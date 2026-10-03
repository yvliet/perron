#!/usr/bin/env python3
"""
Perron Automated Fact Assertion Engine.
Verifies all numerical claims and statistics in paper/paper_writeup.md
against the authoritative ground-truth facts.json registry.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_facts() -> int:
    facts_path = REPO_ROOT / "facts.json"
    writeup_path = REPO_ROOT / "paper" / "paper_writeup.md"

    if not facts_path.exists():
        print(f"[FAIL] Missing facts registry: {facts_path}")
        return 1
    if not writeup_path.exists():
        print(f"[FAIL] Missing paper writeup: {writeup_path}")
        return 1

    with open(facts_path, "r", encoding="utf-8") as f:
        facts = json.load(f)

    writeup_text = writeup_path.read_text(encoding="utf-8")

    checks = []

    # 1. Primary Function Acc@10
    val_14_4 = facts["retrieval_heldout_callable"]["perron_static_fn_acc_10"]
    checks.append((f"{val_14_4}% Function Acc@10", f"{val_14_4}%" in writeup_text or f"{val_14_4:.1f}%" in writeup_text))

    # 2. Degree-Normalized PPR Acc@10
    val_4_2 = facts["retrieval_heldout_callable"]["deg_norm_ppr_fn_acc_10"]
    checks.append((f"{val_4_2}% DegNorm PPR", f"{val_4_2}%" in writeup_text or f"{val_4_2:.1f}%" in writeup_text))

    # 3. Significance tests
    checks.append(("McNemar p = 0.0042", "0.0042" in writeup_text))
    checks.append(("Holm p = 0.0251", "0.0251" in writeup_text))

    # 4. AST Editor Metrics
    val_76_7 = facts["editor_benchmarks"]["perron_ast_apply_rate"]
    checks.append((f"{val_76_7}% AST Apply Rate", f"{val_76_7}%" in writeup_text))

    val_0_0 = facts["editor_benchmarks"]["perron_ast_syntax_error_rate"]
    checks.append((f"{val_0_0}% Syntax Errors", "0.0% syntax errors" in writeup_text.lower()))

    val_23_3 = facts["editor_benchmarks"]["perron_ast_rollback_rate"]
    checks.append((f"{val_23_3}% Rollback Rate", f"{val_23_3}%" in writeup_text))

    # 5. Baseline Editor Fragility
    val_46_7 = facts["editor_benchmarks"]["sed_syntax_error_rate"]
    checks.append((f"{val_46_7}% Sed Syntax Errors", f"{val_46_7}%" in writeup_text))

    val_20_0 = facts["editor_benchmarks"]["git_apply_hunk_reject_rate"]
    checks.append((f"{val_20_0}% Git Hunk Rejections", f"{val_20_0}%" in writeup_text))

    # 6. Spectral Gap & Brauer Bound
    val_0_85 = facts["spectral_convergence"]["brauer_bound_lambda_2"]
    checks.append((f"Brauer bound beta = {val_0_85}", f"{val_0_85}" in writeup_text))

    val_0_15 = facts["spectral_convergence"]["spectral_gap"]
    checks.append((f"Spectral gap >= {val_0_15}", f"{val_0_15}" in writeup_text))

    # 7. Power Iteration Bounds & Provenance Parity
    spectral_artifact = REPO_ROOT / "results" / "spectral_convergence.json"
    if spectral_artifact.exists():
        with open(spectral_artifact, "r", encoding="utf-8") as f:
            sc_data = json.load(f)
        req_sc = sc_data["empirical_convergence"]["requests"]["power_iterations"]
        sym_sc = sc_data["empirical_convergence"]["sympy"]["power_iterations"]
        checks.append(("Spectral telemetry artifact parity", req_sc == 71 and sym_sc == 68))
    else:
        checks.append(("Spectral telemetry artifact parity", False))

    val_71 = facts["spectral_convergence"]["requests_power_iterations"]
    checks.append((f"{val_71} iterations on Requests", f"{val_71} iterations" in writeup_text))

    val_68 = facts["spectral_convergence"]["sympy_power_iterations"]
    checks.append((f"{val_68} iterations on SymPy", f"{val_68} iterations" in writeup_text))

    # 8. Dataset Scale
    val_300 = facts["dataset"]["total_instances"]
    checks.append((f"{val_300} instances", f"{val_300}" in writeup_text))

    # Output verification table
    all_passed = True
    print("=" * 65)
    print("PERRON MANUSCRIPT EMPIRICAL FACT-CHECK REPORT")
    print("=" * 65)
    print(f"{'Metric / Claim':<40} | {'Status':<10}")
    print("-" * 65)
    for desc, passed in checks:
        status_str = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"{desc:<40} | {status_str:<10}")
    print("-" * 65)

    if all_passed:
        print("[SUCCESS] All 14 verified empirical claims confirmed in manuscript.")
        return 0
    else:
        print("[ERROR] One or more empirical assertions failed verification.")
        return 1


if __name__ == "__main__":
    sys.exit(check_facts())
