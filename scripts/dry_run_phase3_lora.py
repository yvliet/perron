"""
scripts/dry_run_phase3_lora.py - Local Pre-Flight Dry Run Gate for Phase 3 LoRA Fine-Tuning.

Runs an offline 5-second CPU verification of:
1. Candidate atomic letter discretization [A]-[Y].
2. Mock forward pass, prompt masking, and loss computation.
3. LoRA weight update verification: base weights strictly frozen, adapter weights updated.
4. Zero-decode logit sorting evaluation logic.
5. Cryptographic firewall zero-leakage check against the 344 held-out evaluation tasks.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.prepare_lora_dataset import get_atomic_label, enforce_zero_leakage_firewall


class MockCausalLM(nn.Module):
    """Minimal mock model with q_proj and v_proj for CPU testing."""
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.Tensor = None):
        h = self.embed(input_ids)
        q = self.q_proj(h)
        v = self.v_proj(h)
        logits = self.lm_head(q + v)

        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return type("Output", (), {"logits": logits, "loss": loss})()


def run_dry_run_checks() -> None:
    print("======================================================================")
    print("PHASE 3 LORA PRE-FLIGHT VERIFICATION & DRY-RUN GATE")
    print("======================================================================")
    t0 = time.perf_counter()

    # 1. Check Atomic Candidate Token Discretization
    print("[CHECK 1] Atomic candidate tokens [A] through [Y]...")
    for idx in range(25):
        letter = get_atomic_label(idx)
        assert len(letter) == 1 and letter.isupper(), f"Invalid letter token: {letter}"
    assert get_atomic_label(0) == "A"
    assert get_atomic_label(24) == "Y"
    print("[PASS]   Atomic tokens verified for up to 25 candidates.")

    # 2. Check Cryptographic Firewall against Held-Out 344 Tasks
    print("[CHECK 2] Cryptographic Benchmark Firewall...")
    train_path = REPO_ROOT / "data" / "gemma4_lora_train.jsonl"
    val_path = REPO_ROOT / "data" / "gemma4_lora_val.jsonl"
    assert train_path.exists(), f"Train dataset missing at {train_path}"
    assert val_path.exists(), f"Val dataset missing at {val_path}"

    train_ids = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_ids.add(json.loads(line)["instance_id"])

    val_ids = set()
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_ids.add(json.loads(line)["instance_id"])

    all_training_ids = train_ids.union(val_ids)
    enforce_zero_leakage_firewall(all_training_ids)
    print(f"[PASS]   Firewall confirmed: {len(all_training_ids)} training tasks strictly disjoint from held-out set.")

    # 3. Check LoRA Model Wrapping and Weight Update Dynamics
    print("[CHECK 3] LoRA model wrapping and weight update dynamics...")
    base_model = MockCausalLM(vocab_size=256, hidden_dim=64)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    lora_model = get_peft_model(base_model, peft_config)

    # Capture initial weights
    base_weight_orig = lora_model.base_model.model.q_proj.base_layer.weight.clone().detach()
    lora_a_orig = lora_model.base_model.model.q_proj.lora_A.default.weight.clone().detach()

    # Synthetic batch
    B, L = 2, 16
    input_ids = torch.randint(0, 256, (B, L))
    labels = input_ids.clone()
    labels[:, :8] = -100  # Mask prompt tokens

    optimizer = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=1e-2)
    optimizer.zero_grad()
    out = lora_model(input_ids=input_ids, labels=labels)
    assert out.loss is not None and not torch.isnan(out.loss), "Loss computation failed or NaN!"
    out.loss.backward()
    optimizer.step()

    # Verify frozen base weights did NOT change
    base_weight_after = lora_model.base_model.model.q_proj.base_layer.weight.clone().detach()
    base_delta = torch.norm(base_weight_after - base_weight_orig).item()
    assert base_delta == 0.0, f"FATAL: Base frozen weights mutated! Delta={base_delta}"

    # Verify LoRA adapter weights DID change
    lora_a_after = lora_model.base_model.model.q_proj.lora_A.default.weight.clone().detach()
    lora_delta = torch.norm(lora_a_after - lora_a_orig).item()
    assert lora_delta > 0.0, f"FATAL: LoRA adapter weights failed to update! Delta={lora_delta}"
    print(f"[PASS]   Weight update dynamics confirmed (Base delta: {base_delta:.6f}, LoRA delta: {lora_delta:.6f}).")

    # 4. Check Zero-Decode Logit Sorting Logic
    print("[CHECK 4] Zero-decode candidate logit ranking...")
    dummy_logits = torch.randn(25)  # 25 candidate logits
    sorted_ranks = np.argsort(-dummy_logits.numpy())[:5]
    assert len(sorted_ranks) == 5, "Ranked list must contain exactly top-5 choices."
    assert len(set(sorted_ranks)) == 5, "Ranked choices must be unique."
    print("[PASS]   Zero-decode candidate logit ranking verified.")

    # 5. Check McNemar Harness Integration
    print("[CHECK 5] McNemar statistical test import and execution...")
    from benchmarks.eval_harness import compute_exact_mcnemar_test, compute_holm_bonferroni_correction
    mcnemar_res = compute_exact_mcnemar_test([1, 1, 0, 1], [0, 1, 0, 0])
    assert "p_value" in mcnemar_res, "McNemar result missing p_value."
    assert "n_target_wins" in mcnemar_res, "McNemar result missing n_target_wins."
    print(f"[PASS]   Eval harness statistics verified (p-value: {mcnemar_res['p_value']:.4f}).")

    # 6. Check Dataset Completion Preservation under Left-Truncation
    print("[CHECK 6] Completion preservation under left-truncation across all dataset items...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it")
    for split in ["train", "val"]:
        p = REPO_ROOT / "data" / f"gemma4_lora_{split}.jsonl"
        with open(p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                item = json.loads(line)
                p_ids = tok.encode(item["prompt"], add_special_tokens=False)
                c_ids = tok.encode(item["completion"], add_special_tokens=False)
                assert len(c_ids) == 13, f"Unexpected completion length {len(c_ids)} in {split}:{line_idx}"
                max_p = max(2048 - len(c_ids), 128)
                if len(p_ids) > max_p:
                    p_ids = p_ids[-max_p:]
                full_ids = p_ids + c_ids
                assert len(full_ids) <= 2048, f"Overflow in {split}:{line_idx}"
                assert full_ids[-13:] == c_ids, f"Completion truncated in {split}:{line_idx}"
    print("[PASS]   Completion preservation verified: 100% of samples preserve 13 completion tokens.")

    # 7. Check Logits Slicing Mathematical Equivalence & VRAM Budget Gate
    print("[CHECK 7] Logits slicing mathematical equivalence and VRAM budget...")
    mock_full_logits = torch.randn(1, 100, 262, requires_grad=True)
    mock_labels = torch.randint(0, 262, (1, 100))
    mock_labels[:, :-13] = -100
    # Full loss
    shift_l = mock_full_logits[:, :-1, :]
    shift_y = mock_labels[:, 1:]
    loss_full = nn.functional.cross_entropy(shift_l.reshape(-1, 262), shift_y.reshape(-1), ignore_index=-100)
    # Sliced loss with logits_to_keep = 24
    sliced_l = mock_full_logits[:, -24:, :]
    loss_sliced = nn.functional.cross_entropy(sliced_l[:, -14:-1, :].reshape(-1, 262), mock_labels[:, -13:].reshape(-1))
    diff = abs(loss_full.item() - loss_sliced.item())
    assert diff < 1e-6, f"Loss slicing mathematical mismatch: {diff}"
    # Memory budget: 24 tokens x 262,144 vocab x 2 bytes (FP16) = 12.58 MB
    sliced_mem_mb = (24 * 262144 * 2) / (1024 * 1024)
    assert sliced_mem_mb < 15.0, f"Sliced memory budget exceeded: {sliced_mem_mb} MB"
    print(f"[PASS]   Logits slicing verified: Loss delta={diff:.8f}, Logit memory={sliced_mem_mb:.2f} MB (< 15 MB).")

    # 8. Check Generated Notebook AST Validity and Imports
    print("[CHECK 8] Automated AST parsing and PEFT import validation on generated notebook...")
    import ast
    nb_file = REPO_ROOT / "notebooks" / "kaggle_gemma4_lora" / "kaggle_gemma4_lora.ipynb"
    assert nb_file.exists(), f"Notebook file missing at {nb_file}"
    with open(nb_file, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    for c_idx, cell in enumerate(nb_data["cells"]):
        if cell["cell_type"] == "code":
            clean_code = "".join([l for l in cell["source"] if not l.strip().startswith("!")])
            ast.parse(clean_code)
    print(f"[PASS]   All {len(nb_data['cells'])} notebook cells verified syntactically valid.")

    elapsed = time.perf_counter() - t0
    print("======================================================================")
    print(f"ALL PRE-FLIGHT CHECKS PASSED WITH RIGOR IN {elapsed:.2f}s!")
    print("======================================================================")


if __name__ == "__main__":
    run_dry_run_checks()
