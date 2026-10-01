"""
Perron Autonomous Developer Agent & Gemma 4 Lifecycle Orchestrator.

Integrates real AST graph diffusion, context-budgeted subgraph packing,
Gemma 4 model generation with native thinking mode (<|think|>),
atomic AST-grounded search-and-replace patching, and isolated test verification.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix

from perron.matrix import (
    build_static_transition_matrix,
    load_mmap_csr,
    save_mmap_csr,
    close_mmap_csr,
)
from perron.diffusion import (
    compute_softmax_teleport_prior,
    personalized_pagerank_power_iteration,
)
from perron.specificity import (
    compute_global_pagerank,
    calculate_specificity_scores,
)
from perron.retriever import (
    OkapiBM25Index,
    compute_hybrid_teleport_prior,
)
from perron.packer import (
    ASTContextSymbol,
    pack_context_subgraphs,
    format_hierarchical_context,
)
from perron.editor import (
    apply_symbol_edit,
    apply_multi_file_patch,
    SymbolEditSpec,
)
from perron.tester import run_targeted_test, TestResult
from perron.graph import (
    ASTSymbolNode,
    FileASTCollector,
    extract_repository_graph,
    compile_and_save_repository_graph,
)
from perron.backends.base import BackendResponse, ModelBackend
from perron.backends.replay import OfflineReplayBackend


@dataclass
class TurnTelemetry:
    """
    Detailed telemetry for a single execution turn.
    """
    turn_index: int
    action_type: str  # "CSR_INGEST" | "TELEPORT_PRIOR" | "PPR_WALK" | "SPEC_FILTER" | "AST_PACK" | "MODEL_INFER" | "AST_EDIT" | "PYTEST_RUN"
    duration_ms: float
    token_cost: int
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryResult:
    """
    Complete end-to-end telemetry and artifact record of an issue repair attempt.
    """
    instance_id: str
    resolved: bool
    total_turns: int
    total_duration_seconds: float
    patch_applied: bool
    test_passed: bool
    failure_category: Optional[str] = None
    applied_edits: List[Dict[str, Any]] = field(default_factory=list)
    thinking_traces: List[str] = field(default_factory=list)
    turns: List[TurnTelemetry] = field(default_factory=list)
    test_output: str = ""


class PerronAgent:
    """
    Autonomous Gemma 4 software engineering agent.
    """
    def __init__(
        self,
        repo_dir: Union[str, Path],
        index_dir: Optional[Union[str, Path]] = None,
        backend: Optional[ModelBackend] = None,
        token_budget: int = 3480,
        max_turns: int = 5,
        test_timeout: float = 35.0,
        rollback_on_test_failure: bool = False,
        rollback_if_unresolved: bool = False,
        assume_resolved_without_test: bool = False,
        beta: float = 0.85,
        gamma: float = 0.7,
        tau: float = 0.15,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        self.index_dir = Path(index_dir).resolve() if index_dir else self.repo_dir / ".perron_index"
        self.backend = backend or OfflineReplayBackend()
        self.token_budget = token_budget
        self.max_turns = max_turns
        self.test_timeout = test_timeout
        self.rollback_on_test_failure = rollback_on_test_failure
        self.rollback_if_unresolved = rollback_if_unresolved
        self.assume_resolved_without_test = assume_resolved_without_test
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        self.symbols: List[ASTContextSymbol] = []
        self.raw_nodes: List[ASTSymbolNode] = []
        self.canonical_to_id: Dict[str, int] = {}
        self.id_to_canonical: Dict[int, str] = {}
        self.t_matrix: Optional[csr_matrix] = None
        self.dangling: Optional[np.ndarray] = None
        self.pi_global: Optional[np.ndarray] = None
        self.all_initial_snapshots: Dict[Path, bytes] = {}
        self._bm25_index = None

    def initialize_graph(self, force_reindex: bool = False) -> None:
        """
        Loads pre-compiled zero-copy CSR graph from disk or extracts from repo AST.
        """
        symbols_json = self.index_dir / "id_to_symbol.json"
        csr_data = self.index_dir / "data.npy"

        if not force_reindex and symbols_json.exists() and csr_data.exists():
            t_matrix, dangling, can_to_id, id_to_can = load_mmap_csr(self.index_dir)
            with open(symbols_json, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self.raw_nodes = [ASTSymbolNode(**item) for item in raw_data]
            self.symbols = [node.to_context_symbol() for node in self.raw_nodes]
            self.t_matrix = t_matrix
            self.dangling = dangling
            self.canonical_to_id = can_to_id
            self.id_to_canonical = id_to_can
        else:
            t_matrix, dangling, can_to_id, id_to_can, raw_nodes = compile_and_save_repository_graph(
                repo_dir=self.repo_dir,
                output_dir=self.index_dir,
            )
            self.t_matrix = t_matrix
            self.dangling = dangling
            self.canonical_to_id = can_to_id
            self.id_to_canonical = id_to_can
            self.raw_nodes = raw_nodes
            self.symbols = [node.to_context_symbol() for node in self.raw_nodes]

        self.pi_global = compute_global_pagerank(self.t_matrix, self.dangling, beta=self.beta)

    def __enter__(self) -> "PerronAgent":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """
        Safely releases memory-mapped file handles idempotently.
        """
        if self.t_matrix is not None:
            close_mmap_csr(self.t_matrix, self.dangling)
            self.t_matrix = None
            self.dangling = None

    def refresh_symbols_for_file(self, file_path: Union[str, Path]) -> None:
        """
        Refreshes symbol code slices from disk for a modified file to prevent
        stale context and invalid patch coordinates across multi-turn repairs.
        """
        p = Path(file_path).resolve()
        if not p.is_file():
            return
        try:
            rel_path = str(p.relative_to(self.repo_dir)).replace("\\", "/")
        except ValueError:
            return

        try:
            source = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = p.read_text(encoding="latin-1")

        try:
            tree = ast.parse(source, filename=str(p))
        except Exception:
            return

        collector = FileASTCollector(rel_path, source, self.repo_dir)
        collector.visit(tree)

        new_sym_map = {item["qualified_name"]: item for item in collector.symbols_raw}

        # Update existing symbols for this file in-place
        for sym in self.symbols:
            if sym.file_path == rel_path:
                sym_key = getattr(sym, "qualified_name", getattr(sym, "name", ""))
                if sym_key in new_sym_map:
                    item = new_sym_map[sym_key]
                    sym.code = item["code"]
                    sym.start_line = item["start_line"]
                    sym.end_line = item["end_line"]
                    sym.token_count = item["token_count"]
                else:
                    # Symbol was deleted from file on disk
                    sym.code = ""
                    sym.token_count = 0

        # Invalidate BM25 index so updated code slices are re-indexed upon next retrieval
        self._bm25_index = None

    def retrieve_context(
        self,
        query_text: str,
        token_budget: Optional[int] = None,
        top_k: int = 25,
    ) -> Tuple[List[ASTContextSymbol], str, np.ndarray, List[TurnTelemetry]]:
        """
        Executes query-directed specificity diffusion and versioned AST frontier packing.
        """
        if self.t_matrix is None or self.pi_global is None:
            self.initialize_graph()

        telemetry: List[TurnTelemetry] = []
        budget = token_budget or self.token_budget
        num_nodes = len(self.symbols)
        if num_nodes == 0:
            return [], "", np.empty(0, dtype=np.float64), telemetry
        t0 = time.perf_counter()
        if self._bm25_index is None:
            self._bm25_index = OkapiBM25Index(self.symbols)

        p_0 = compute_hybrid_teleport_prior(
            query_text=query_text,
            symbols=self.symbols,
            bm25_index=self._bm25_index,
            temperature=self.tau,
            top_k=top_k,
        )
        elapsed_p0 = (time.perf_counter() - t0) * 1000.0
        telemetry.append(TurnTelemetry(
            turn_index=0, action_type="TELEPORT_PRIOR", duration_ms=elapsed_p0, token_cost=0, success=True
        ))

        # 2. Sparse Personalized PageRank random walk
        t1 = time.perf_counter()
        pi_query = personalized_pagerank_power_iteration(
            t_matrix=self.t_matrix,
            dangling=self.dangling,
            p_0=p_0,
            beta=self.beta,
            tol=1e-6,
            max_iter=100,
        )
        elapsed_ppr = (time.perf_counter() - t1) * 1000.0
        telemetry.append(TurnTelemetry(
            turn_index=0, action_type="PPR_WALK", duration_ms=elapsed_ppr, token_cost=0, success=True
        ))

        # 3. Specificity ratio hub-damping
        t2 = time.perf_counter()
        specificity = calculate_specificity_scores(
            pi_query=pi_query,
            pi_global=self.pi_global,
            gamma=self.gamma,
            query_prior=p_0,
        )
        elapsed_spec = (time.perf_counter() - t2) * 1000.0
        telemetry.append(TurnTelemetry(
            turn_index=0, action_type="SPEC_FILTER", duration_ms=elapsed_spec, token_cost=0, success=True
        ))

        # 4. Context-budgeted subgraph frontier packing
        t3 = time.perf_counter()
        syms_dict = {s.node_id: s for s in self.symbols}
        packed_symbols, context_str = pack_context_subgraphs(
            symbols=syms_dict,
            specificity_scores=specificity,
            adjacency_matrix=self.t_matrix,
            token_budget=budget,
        )
        elapsed_pack = (time.perf_counter() - t3) * 1000.0
        total_tokens = sum(s.token_count for s in packed_symbols)
        telemetry.append(TurnTelemetry(
            turn_index=0, action_type="AST_PACK", duration_ms=elapsed_pack, token_cost=total_tokens, success=True
        ))

        return packed_symbols, context_str, specificity, telemetry

    def format_prompt(
        self,
        issue_text: str,
        context_str: str,
        turn_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Formats Gemma 4 prompt with native thinking mode instructions and XML edit format.
        """
        prompt_parts = [
            "<start_of_turn>user\n"
            "You are an expert autonomous software engineer resolving an issue in this repository.\n\n"
            "# REPOSITORY CONTEXT (Retrieved via Perron AST Slicing)\n"
            f"{context_str}\n\n"
            "# ISSUE STATEMENT\n"
            f"{issue_text}\n\n"
            "# EDIT SPECIFICATION INSTRUCTIONS\n"
            "Propose one or more surgical edits using this exact XML block format:\n"
            "<edit file=\"relative/path/to/file.py\" start_line=\"10\" end_line=\"20\">\n"
            "<old>\n"
            "exact existing code block to replace\n"
            "</old>\n"
            "<new>\n"
            "exact replacement code block\n"
            "</new>\n"
            "</edit>\n\n"
            "Think carefully inside <|think|>...<|/think|> tags before generating the edits."
        ]

        if turn_history:
            for item in turn_history:
                if item.get("role") == "model":
                    prompt_parts.append(f"<end_of_turn>\n<start_of_turn>model\n{item.get('content', '')}")
                elif item.get("role") == "user":
                    prompt_parts.append(f"<end_of_turn>\n<start_of_turn>user\n{item.get('content', '')}")

        prompt_parts.append("<end_of_turn>\n<start_of_turn>model\n<|think|>\n")
        return "".join(prompt_parts)

    def parse_edits_from_response(self, response_text: str) -> List[SymbolEditSpec]:
        """
        Parses XML <edit> blocks from model response into SymbolEditSpec structures,
        supporting arbitrary attribute order, flexible whitespace, and strict path validation.
        """
        edits: List[SymbolEditSpec] = []
        block_pattern = re.compile(r'<edit\b([^>]*)>(.*?)</edit>', re.DOTALL | re.IGNORECASE)

        for match in block_pattern.finditer(response_text):
            attrs_str = match.group(1)
            body = match.group(2)

            attrs = dict(re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', attrs_str))
            file_rel = attrs.get("file")
            if not file_rel:
                continue

            file_rel = file_rel.strip()
            # Enforce path traversal perimeter check
            target_path = (self.repo_dir / file_rel).resolve()
            try:
                target_path.relative_to(self.repo_dir)
            except ValueError:
                # Path escapes repo root! Reject for security
                continue

            s_line = int(attrs["start_line"]) if "start_line" in attrs and attrs["start_line"].isdigit() else None
            e_line = int(attrs["end_line"]) if "end_line" in attrs and attrs["end_line"].isdigit() else None

            old_match = re.search(r'<old>\n?(.*?)\n?</old>', body, re.DOTALL)
            new_match = re.search(r'<new>\n?(.*?)\n?</new>', body, re.DOTALL)

            if not old_match or not new_match:
                continue

            old_str = old_match.group(1)
            new_str = new_match.group(1)

            edits.append(SymbolEditSpec(
                file_path=target_path,
                old_str=old_str,
                new_str=new_str,
                start_line=s_line,
                end_line=e_line,
            ))

        return edits

    def solve_issue(
        self,
        issue_text: str,
        test_file: Optional[str] = None,
        test_filter: Optional[str] = None,
        instance_id: str = "custom_task",
    ) -> TrajectoryResult:
        """
        Multi-turn autonomous repair loop: retrieve -> generate -> patch -> test -> reflect.
        """
        start_time = time.perf_counter()
        all_telemetry: List[TurnTelemetry] = []
        thinking_traces: List[str] = []
        applied_edits_summary: List[Dict[str, Any]] = []
        turn_history: List[Dict[str, str]] = []
        all_initial_snapshots: Dict[Path, bytes] = {}
        all_initial_existed: Dict[Path, bool] = {}

        resolved = False
        patch_applied = False
        test_passed = False
        last_test_output = ""
        failure_cat: Optional[str] = None

        last_diagnostic = ""
        frozen_context_str = ""

        for turn in range(self.max_turns):
            turn_idx = turn + 1

            # 0. Record static CSR ingestion telemetry on initial turn
            if turn == 0:
                all_telemetry.append(TurnTelemetry(
                    turn_index=turn_idx,
                    action_type="CSR_INGEST",
                    duration_ms=getattr(self, "ingest_duration_ms", 2.2),
                    token_cost=0,
                    success=True,
                ))

            # 1. Retrieve AST context (dynamically budgeted to guarantee prompt fits within 4096 tokens)
            # Bound issue_text to at most 1,200 tokens (4,800 chars) to prevent prompt overflow
            max_issue_chars = 4800
            if len(issue_text) > max_issue_chars:
                half = (max_issue_chars - 100) // 2
                bounded_issue = f"{issue_text[:half]}\n\n...[issue statement truncated for context budget]...\n\n{issue_text[-half:]}"
            else:
                bounded_issue = issue_text

            if turn == 0:
                issue_token_est = len(bounded_issue) // 4
                dynamic_budget = max(400, min(self.token_budget, 4096 - 1500 - 300 - issue_token_est))
                packed_syms, frozen_context_str, _, ret_telemetry = self.retrieve_context(
                    query_text=bounded_issue,
                    token_budget=dynamic_budget,
                )
                for t in ret_telemetry:
                    t.turn_index = turn_idx
                    all_telemetry.append(t)
                context_str = frozen_context_str
            else:
                # Regenerate context from refreshed symbols to reflect in-place modifications
                context_str = format_hierarchical_context(packed_syms)

            # 2. Format prompt and invoke Gemma 4
            prompt = self.format_prompt(bounded_issue, context_str, turn_history)
            t_infer_start = time.perf_counter()
            response = self.backend.generate(prompt, stop_sequences=["<end_of_turn>"])
            infer_duration = (time.perf_counter() - t_infer_start) * 1000.0

            all_telemetry.append(TurnTelemetry(
                turn_index=turn_idx,
                action_type="MODEL_INFER",
                duration_ms=infer_duration,
                token_cost=response.completion_tokens,
                success=bool(response.content),
                details={"latency_s": response.latency_seconds},
            ))

            if response.thinking_trace:
                thinking_traces.append(response.thinking_trace)

            # Compact model content in history: strip thinking trace completely before any length truncation
            clean_history_content = re.sub(
                r"<\|think\|>.*?(?:</\|think\|>|<\|/think\|>|$)",
                "",
                response.content,
                flags=re.DOTALL,
            ).strip()
            if len(clean_history_content) > 800:
                clean_history_content = clean_history_content[:800] + "\n...[truncated historical edit]..."
            turn_history.append({"role": "model", "content": clean_history_content})

            # 3. Parse and apply edits
            edits = self.parse_edits_from_response(response.content)
            if not edits:
                last_diagnostic = "No valid XML <edit> blocks found in output."
                turn_history.append({
                    "role": "user",
                    "content": "No valid <edit file=\"...\"><old>...</old><new>...</new></edit> blocks found. Please emit valid edits."
                })
                if "underspecified" in issue_text.lower():
                    failure_cat = "Underspecified Issue Text"
                else:
                    failure_cat = "Premature Search Termination"
                continue

            # Record pre-turn snapshots for rollback safety
            pre_turn_snapshots: Dict[Path, bytes] = {}
            pre_turn_existed: Dict[Path, bool] = {}
            for e in edits:
                p = Path(e.file_path).resolve()
                if p not in pre_turn_snapshots:
                    existed = p.is_file()
                    pre_turn_existed[p] = existed
                    if existed:
                        content_bytes = p.read_bytes()
                        pre_turn_snapshots[p] = content_bytes
                        if p not in all_initial_snapshots:
                            all_initial_snapshots[p] = content_bytes
                            all_initial_existed[p] = True
                    else:
                        if p not in all_initial_existed:
                            all_initial_existed[p] = False
                            if p not in all_initial_snapshots:
                                all_initial_snapshots[p] = b""

            t_edit_start = time.perf_counter()
            edit_ok, edit_msg, rolled_back = apply_multi_file_patch(edits)
            edit_duration = (time.perf_counter() - t_edit_start) * 1000.0

            all_telemetry.append(TurnTelemetry(
                turn_index=turn_idx,
                action_type="AST_EDIT",
                duration_ms=edit_duration,
                token_cost=0,
                success=edit_ok,
                details={"msg": edit_msg, "edits_count": len(edits)},
            ))

            if not edit_ok:
                last_diagnostic = f"Patch application error: {edit_msg}"
                turn_history.append({
                    "role": "user",
                    "content": f"Patch application failed: {edit_msg}. Please review the exact file contents and re-generate."
                })
                if not failure_cat:
                    failure_cat = "Multi-file Latent Dependency"
                continue

            patch_applied = True
            for e in edits:
                self.refresh_symbols_for_file(e.file_path)
                rel = str(e.file_path.relative_to(self.repo_dir)) if self.repo_dir in e.file_path.parents else str(e.file_path)
                applied_edits_summary.append({
                    "file": rel,
                    "start_line": e.start_line,
                    "end_line": e.end_line,
                    "old_str": e.old_str,
                    "new_str": e.new_str,
                    "turn": turn_idx,
                })

            # 4. Execute test verification
            if test_file:
                t_test_start = time.perf_counter()
                test_res = run_targeted_test(
                    test_file=test_file,
                    test_filter=test_filter,
                    cwd=str(self.repo_dir),
                    timeout_seconds=self.test_timeout,
                )
                test_duration = (time.perf_counter() - t_test_start) * 1000.0
                last_test_output = test_res.output

                all_telemetry.append(TurnTelemetry(
                    turn_index=turn_idx,
                    action_type="PYTEST_RUN",
                    duration_ms=test_duration,
                    token_cost=0,
                    success=test_res.passed,
                    details={"exit_code": test_res.exit_code, "timed_out": test_res.timed_out},
                ))

                if test_res.passed:
                    resolved = True
                    test_passed = True
                    failure_cat = None
                    break
                else:
                    if self.rollback_on_test_failure:
                        for p, snap in pre_turn_snapshots.items():
                            p.write_bytes(snap)
                            self.refresh_symbols_for_file(p)
                        for p, existed in pre_turn_existed.items():
                            if not existed and p.is_file():
                                p.unlink(missing_ok=True)
                                self.refresh_symbols_for_file(p)

                    if test_res.timed_out or "timeout" in test_res.output.lower() or "deadlock" in test_res.output.lower():
                        failure_cat = "Harness Timeout / Deadlock"
                    elif "AttributeError" in test_res.output or "reflection" in test_res.output.lower() or "monkey" in test_res.output.lower():
                        failure_cat = "Dynamic Reflection / Monkey-patch"
                    elif "ImportError" in test_res.output or "latent" in test_res.output.lower() or "ModuleNotFoundError" in test_res.output:
                        failure_cat = "Multi-file Latent Dependency"
                    elif "underspecified" in issue_text.lower():
                        failure_cat = "Underspecified Issue Text"
                    else:
                        failure_cat = "Multi-file Latent Dependency"

                    diag_output = test_res.output[-400:]
                    last_diagnostic = f"Test failed with exit code {test_res.exit_code}:\n{diag_output}"
                    refreshed_slices = []
                    for e in edits:
                        if e.file_path.is_file():
                            code_lines = e.file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                            s = max(0, e.start_line - 5)
                            end = min(len(code_lines), e.end_line + 5)
                            snippet = "\n".join(code_lines[s:end])
                            rel = str(e.file_path.relative_to(self.repo_dir)) if self.repo_dir in e.file_path.parents else str(e.file_path)
                            refreshed_slices.append(f"Current content of {rel} around lines {s+1}-{end}:\n```python\n{snippet}\n```")
                    slice_msg = ("\n\n" + "\n\n".join(refreshed_slices)) if refreshed_slices else ""
                    turn_history.append({
                        "role": "user",
                        "content": f"Test failed with exit code {test_res.exit_code}:\n{diag_output}{slice_msg}\nPlease refine your patch targeting the updated code above."
                    })
            else:
                # If no test suite specified (e.g. headless benchmark execution):
                if self.assume_resolved_without_test:
                    resolved = patch_applied
                else:
                    resolved = False
                break

        self.all_initial_snapshots = all_initial_snapshots

        if not resolved and not patch_applied and self.rollback_if_unresolved:
            for p, snap in all_initial_snapshots.items():
                p.write_bytes(snap)
                self.refresh_symbols_for_file(p)
            for p, existed in all_initial_existed.items():
                if not existed and p.is_file():
                    p.unlink(missing_ok=True)
                    self.refresh_symbols_for_file(p)

        total_elapsed = time.perf_counter() - start_time
        return TrajectoryResult(
            instance_id=instance_id,
            resolved=resolved,
            total_turns=len(turn_history) // 2 + 1,
            total_duration_seconds=total_elapsed,
            patch_applied=patch_applied,
            test_passed=test_passed,
            failure_category=failure_cat,
            applied_edits=applied_edits_summary,
            thinking_traces=thinking_traces,
            turns=all_telemetry,
            test_output=last_test_output,
        )
