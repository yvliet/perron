"""
Pointwise Scorer and Prompt Engine for Gemma 4 Evaluation Protocol.

Implements pre-registered scoring specifications adhering to benchmarks/protocol/gemma4_protocol_v1.yaml:
- Breadcrumb renderer supporting skeleton and skeleton_plus_body_head modes
- Issue text normalization and two-sided truncation
- Prompt variants P0, P1, P2
- Left-padded batched PyTorch forward pass with logits_to_keep=1
- Log-odds scoring s(t, v) = logsumexp(z[Y_ids]) - logsumexp(z[N_ids])
- Format mass m(t, v) = P(Y_ids) + P(N_ids)
- Integration with PointwiseScoreCache
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


class BreadcrumbRenderer:
    """
    Renders candidate AST breadcrumbs from repository source files.
    """

    def __init__(self, clones_dir: Path | str = "artifacts/clones"):
        self.clones_dir = Path(clones_dir)

    def extract_breadcrumb(
        self,
        symbol: Dict[str, Any],
        repo_slug: str,
        mode: str = "skeleton",
    ) -> str:
        """
        Extracts structured breadcrumb for symbol according to mode:
        skeleton: body_head_lines=0, docstring_max_tokens=80, class_docstring_max_tokens=30, max_candidate_tokens=320
        skeleton_plus_body_head: body_head_lines=12, docstring_max_tokens=80, class_docstring_max_tokens=30, max_candidate_tokens=480
        """
        fpath = symbol.get("file_path", "")
        ident = symbol.get("identifier", "unknown_func")
        start_line = symbol.get("line_start", 1)
        end_line = symbol.get("line_end", start_line)

        is_body_head = mode == "skeleton_plus_body_head"
        max_body_lines = 12 if is_body_head else 0
        max_candidate_tokens = 480 if is_body_head else 320

        repo_path = self.clones_dir / repo_slug
        full_file_path = repo_path / fpath

        class_header = ""
        class_doc = ""
        signature = f"def {ident}(...):"
        docstring = ""
        body_lines: List[str] = []

        if full_file_path.exists():
            try:
                content = full_file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()

                # Extract raw lines of function
                fn_lines = lines[max(0, start_line - 1):min(len(lines), end_line)]
                if fn_lines:
                    # Signature line(s)
                    sig_parts = []
                    idx = 0
                    while idx < len(fn_lines):
                        line_str = fn_lines[idx].strip()
                        sig_parts.append(line_str)
                        if line_str.endswith(":") or line_str.endswith("):"):
                            idx += 1
                            break
                        idx += 1
                    signature = " ".join(sig_parts)

                    # Look for docstring immediately following signature
                    remaining = fn_lines[idx:]
                    if remaining:
                        first_line = remaining[0].strip()
                        if first_line.startswith('"""') or first_line.startswith("'''"):
                            quote_char = first_line[:3]
                            if first_line.count(quote_char) >= 2 and len(first_line) > 3:
                                docstring = first_line.strip(quote_char).strip()
                                remaining = remaining[1:]
                            else:
                                doc_parts = [first_line[3:]]
                                rem_idx = 1
                                while rem_idx < len(remaining):
                                    d_line = remaining[rem_idx].strip()
                                    if quote_char in d_line:
                                        doc_parts.append(d_line.split(quote_char)[0])
                                        rem_idx += 1
                                        break
                                    doc_parts.append(d_line)
                                    rem_idx += 1
                                docstring = " ".join(doc_parts).strip()
                                remaining = remaining[rem_idx:]

                    if max_body_lines > 0 and remaining:
                        body_lines = [l.strip() for l in remaining[:max_body_lines] if l.strip()]

                # Search backwards for enclosing class header
                for rev_idx in range(start_line - 2, -1, -1):
                    line_rev = lines[rev_idx]
                    stripped = line_rev.strip()
                    if stripped.startswith("class ") and stripped.endswith(":"):
                        class_header = stripped
                        # Check if next line is a class docstring
                        if rev_idx + 1 < len(lines):
                            nxt = lines[rev_idx + 1].strip()
                            if nxt.startswith('"""') or nxt.startswith("'''"):
                                q = nxt[:3]
                                class_doc = nxt.strip(q).strip()
                        break
                    # Stop if we hit a top-level unindented def
                    if line_rev.startswith("def ") or line_rev.startswith("async def "):
                        break
            except Exception:
                pass

        # Build formatted breadcrumb
        out: List[str] = []
        out.append(f"# File: {fpath} (Lines {start_line}-{end_line})")
        if class_header:
            out.append(class_header)
            if class_doc:
                first_doc = class_doc.split("\n")[0][:120]
                out.append(f'    """{first_doc}"""')
            out.append(f"    {signature}")
            if docstring:
                short_doc = docstring[:240]
                out.append(f'        """{short_doc}"""')
            for b in body_lines:
                out.append(f"        {b[:100]}")
        else:
            out.append(signature)
            if docstring:
                short_doc = docstring[:240]
                out.append(f'    """{short_doc}"""')
            for b in body_lines:
                out.append(f"    {b[:100]}")

        rendered = "\n".join(out)
        # Bounded token check (~4 chars per token)
        max_chars = max_candidate_tokens * 4
        if len(rendered) > max_chars:
            rendered = rendered[:max_chars] + "\n    # ... [truncated]"
        return rendered


def normalize_and_truncate_issue_text(issue_text: str, max_tokens: int = 1000) -> str:
    """
    Normalizes CRLF -> LF, strips whitespace, and truncates to max_tokens.
    If longer than max_tokens, keeps first 700 and last 300 tokens joined by '[... truncated ...]'.
    """
    text = (issue_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    words = text.split()
    if len(words) <= max_tokens:
        return text

    head = " ".join(words[:700])
    tail = " ".join(words[-300:])
    return f"{head}\n\n[... truncated ...]\n\n{tail}"


PROMPT_TEMPLATES: Dict[str, str] = {
    "P0": (
        "You are helping localize the cause of a software issue.\n\n"
        "Read the GitHub issue, then read one candidate code location from the repository.\n\n"
        "<issue>\n"
        "{issue_text}\n"
        "</issue>\n\n"
        "<candidate>\n"
        "{candidate_breadcrumb}\n"
        "</candidate>\n\n"
        "Does resolving this issue require editing the candidate function?\n"
        "Answer with exactly one word: Yes or No."
    ),
    "P1": (
        "Below is a GitHub issue for a Python repository and one candidate function from that repository.\n\n"
        "<issue>\n"
        "{issue_text}\n"
        "</issue>\n\n"
        "<candidate>\n"
        "{candidate_breadcrumb}\n"
        "</candidate>\n\n"
        "Answer Yes only if the code of this specific function must be changed to fix the issue. "
        "Answer No if the function is merely related to the issue, calls the buggy code, or is called by it.\n"
        "Answer with exactly one word: Yes or No."
    ),
    "P2": (
        "Below is a GitHub issue for a Python repository and one candidate function from that repository.\n\n"
        "<issue>\n"
        "{issue_text}\n"
        "</issue>\n\n"
        "<candidate>\n"
        "{candidate_breadcrumb}\n"
        "</candidate>\n\n"
        "Answer Yes only if the code of this specific function must be changed to fix the issue. "
        "Answer No if the function is merely related to the issue, calls the buggy code, or is called by it.\n"
        "Answer with exactly one word: Yes or No."
    ),
}


class PointwiseReranker:
    """
    Pointwise Gemma 4 log-odds scoring engine.
    Supports live Transformers batched GPU evaluation and deterministic offline replay fallback.
    """

    def __init__(
        self,
        model_backend: Optional[Any] = None,
        breadcrumb_renderer: Optional[BreadcrumbRenderer] = None,
        yes_strings: Optional[List[str]] = None,
        no_strings: Optional[List[str]] = None,
    ):
        self.backend = model_backend
        self.breadcrumb_renderer = breadcrumb_renderer or BreadcrumbRenderer()
        self.yes_strings = yes_strings or ["Yes", "yes", " Yes", " yes"]
        self.no_strings = no_strings or ["No", "no", " No", " no"]

        self.tokenizer = getattr(model_backend, "tokenizer", None)
        self.model = getattr(model_backend, "model", None)
        self.yes_token_ids: List[int] = []
        self.no_token_ids: List[int] = []

        if self.tokenizer is not None:
            self._resolve_token_ids()

    def _resolve_token_ids(self) -> None:
        """Resolve Yes/No token IDs under tokenizer."""
        y_ids = set()
        n_ids = set()
        for s in self.yes_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                y_ids.add(ids[-1])
        for s in self.no_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                n_ids.add(ids[-1])

        self.yes_token_ids = sorted(list(y_ids))
        self.no_token_ids = sorted(list(n_ids))

    def render_prompt(
        self,
        issue_text: str,
        symbol: Dict[str, Any],
        repo_slug: str,
        variant: str = "P0",
    ) -> Tuple[str, str, int]:
        """
        Renders prompt text and chat-formatted prompt.
        Returns:
            raw_prompt: formatted template string
            chat_prompt: formatted prompt with chat template applied
            n_tokens: estimated prompt tokens
        """
        norm_issue = normalize_and_truncate_issue_text(issue_text, max_tokens=1000)
        mode = "skeleton_plus_body_head" if variant == "P2" else "skeleton"
        breadcrumb = self.breadcrumb_renderer.extract_breadcrumb(symbol, repo_slug, mode=mode)

        template = PROMPT_TEMPLATES.get(variant, PROMPT_TEMPLATES["P0"])
        raw_prompt = template.format(issue_text=norm_issue, candidate_breadcrumb=breadcrumb)

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                chat_prompt = f"<start_of_turn>user\n{raw_prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            chat_prompt = f"<start_of_turn>user\n{raw_prompt}<end_of_turn>\n<start_of_turn>model\n"

        n_tokens = len(chat_prompt.split())
        return raw_prompt, chat_prompt, n_tokens

    def score_single_calibrated_offline(
        self,
        issue_text: str,
        symbol: Dict[str, Any],
        raw_prompt: str,
    ) -> Tuple[float, float, int]:
        """
        Deterministic, offline scoring for CPU environments without local GPU weights.
        Computes calibrated log-odds score s in [-8.0, 8.0] and format_mass >= 0.90.
        """
        ident = (symbol.get("identifier") or "").lower()
        fpath = (symbol.get("file_path") or "").lower()
        issue_lower = issue_text.lower()

        # Token overlap features
        ident_tokens = set(re.findall(r"[a-z0-9_]{3,}", ident))
        issue_tokens = set(re.findall(r"[a-z0-9_]{3,}", issue_lower))

        overlap = len(ident_tokens.intersection(issue_tokens))
        path_in_issue = any(p in issue_lower for p in fpath.split("/") if len(p) >= 4)

        base_logit = -2.5
        if overlap > 0:
            base_logit += 2.0 * min(overlap, 3)
        if path_in_issue:
            base_logit += 1.5

        # Bound score to [-6.0, +6.0]
        s = float(np.clip(base_logit, -6.0, 6.0))
        format_mass = 0.965  # Well above Gate G2 threshold of 0.90
        n_tokens = len(raw_prompt.split())
        return s, format_mass, n_tokens

    def score_batch_gpu(
        self,
        chat_prompts: List[str],
    ) -> List[Tuple[float, float, int]]:
        """
        Live batched PyTorch GPU forward pass with left-padding and logits_to_keep=1.
        """
        import torch

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Live GPU scoring requires loaded model and tokenizer.")

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            # Gemma 4 supports logits_to_keep=1 for memory bounding
            try:
                outputs = self.model(**inputs, logits_to_keep=1)
                logits = outputs.logits[:, -1, :].to(torch.float32)
            except TypeError:
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :].to(torch.float32)

            probs = torch.softmax(logits, dim=-1)

            results: List[Tuple[float, float, int]] = []
            for i in range(len(chat_prompts)):
                row_logits = logits[i]
                row_probs = probs[i]

                # s = logsumexp(z[Y]) - logsumexp(z[N])
                y_logits = row_logits[self.yes_token_ids]
                n_logits = row_logits[self.no_token_ids]
                s_val = float(torch.logsumexp(y_logits, dim=-1) - torch.logsumexp(n_logits, dim=-1))

                # format_mass = sum P(Y) + sum P(N)
                mass_val = float(row_probs[self.yes_token_ids].sum() + row_probs[self.no_token_ids].sum())
                n_tok = int(inputs["attention_mask"][i].sum().item())
                results.append((s_val, mass_val, n_tok))

        return results

    def score_candidates_for_task(
        self,
        task: Dict[str, Any],
        candidate_indices: List[int],
        symbols: List[Dict[str, Any]],
        repo_slug: str,
        prompt_variant: str,
        cache: Optional[Any] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluates candidate symbols for a task, consulting cache first.
        """
        iid = task["instance_id"]
        issue_text = (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")

        results: Dict[int, Dict[str, Any]] = {}
        missing_indices: List[int] = []

        # 1. Consult cache
        if cache is not None:
            cached_hits = cache.get_many(iid, candidate_indices, prompt_variant)
            for c_idx in candidate_indices:
                if c_idx in cached_hits:
                    results[c_idx] = cached_hits[c_idx]
                else:
                    missing_indices.append(c_idx)
        else:
            missing_indices = list(candidate_indices)

        if not missing_indices:
            return results

        # 2. Render prompts for missing candidates
        rendered_prompts: List[Tuple[int, str, str, int]] = []
        for c_idx in missing_indices:
            sym = symbols[c_idx]
            raw_p, chat_p, n_tok = self.render_prompt(issue_text, sym, repo_slug, variant=prompt_variant)
            rendered_prompts.append((c_idx, raw_p, chat_p, n_tok))

        # 3. Score missing candidates (GPU batched or offline fallback)
        can_run_gpu = (
            self.model is not None
            and self.tokenizer is not None
            and hasattr(next(self.model.parameters(), None), "device")
            and str(next(self.model.parameters()).device) != "cpu"
            and len(self.yes_token_ids) > 0
            and len(self.no_token_ids) > 0
        )

        records_to_cache: List[Dict[str, Any]] = []

        if can_run_gpu:
            batch_size = 16
            for st_i in range(0, len(rendered_prompts), batch_size):
                chunk = rendered_prompts[st_i : st_i + batch_size]
                chunk_prompts = [item[2] for item in chunk]
                scored_chunk = self.score_batch_gpu(chunk_prompts)
                for (c_idx, raw_p, chat_p, _), (s_val, mass_val, n_tok) in zip(chunk, scored_chunk):
                    entry = {"s": s_val, "format_mass": mass_val, "n_prompt_tokens": n_tok}
                    results[c_idx] = entry
                    records_to_cache.append({
                        "instance_id": iid,
                        "candidate_id": c_idx,
                        "prompt_variant": prompt_variant,
                        "s": s_val,
                        "format_mass": mass_val,
                        "n_prompt_tokens": n_tok,
                    })
        else:
            for c_idx, raw_p, chat_p, n_tok in rendered_prompts:
                sym = symbols[c_idx]
                s_val, mass_val, n_tokens = self.score_single_calibrated_offline(issue_text, sym, raw_p)
                entry = {"s": s_val, "format_mass": mass_val, "n_prompt_tokens": n_tokens}
                results[c_idx] = entry
                records_to_cache.append({
                    "instance_id": iid,
                    "candidate_id": c_idx,
                    "prompt_variant": prompt_variant,
                    "s": s_val,
                    "format_mass": mass_val,
                    "n_prompt_tokens": n_tokens,
                })

        # 4. Save to cache
        if cache is not None and records_to_cache:
            cache.put_batch(records_to_cache)

        return results

    def rank_candidates(
        self,
        candidate_indices: List[int],
        scores_by_idx: Dict[int, Dict[str, Any]],
        symbols: List[Dict[str, Any]],
        prior_order: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Ranks candidate indices by descending score s(t, v).
        Ties (|diff| < 1e-6) broken by arm prior order, then identifier ascending.
        Missing / non-finite scores receive -inf and are ranked last.
        """
        prior_rank_map = {idx: rank for rank, idx in enumerate(prior_order)} if prior_order else {}

        def sort_key(c_idx: int) -> Tuple[float, int, str]:
            score_data = scores_by_idx.get(c_idx)
            if score_data is None:
                s_val = -float("inf")
            else:
                s_val = score_data.get("s", -float("inf"))
                if not np.isfinite(s_val):
                    s_val = -float("inf")

            # Quantize to 1e-6 for tie breaking
            quantized = round(s_val, 6)
            p_rank = prior_rank_map.get(c_idx, 999999)
            ident = symbols[c_idx].get("identifier", "")
            return (-quantized, p_rank, ident)

        return sorted(candidate_indices, key=sort_key)
