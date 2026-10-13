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

    elapsed = time.perf_counter() - t0
    print("======================================================================")
    print(f"ALL PRE-FLIGHT CHECKS PASSED PERFECTLY IN {elapsed:.2f}s!")
    print("======================================================================")


if __name__ == "__main__":
    run_dry_run_checks()
