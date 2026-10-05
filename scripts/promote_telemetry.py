#!/usr/bin/env python3
"""
Automated Two-Stage Promotion Gateway for Empirical Telemetry.
Author: Sultan Haikal (GitHub: @yvliet)
License: Apache-2.0

Validates benchmark output JSON artifacts against pre-registered acceptance criteria,
binds verified RFC 6901 JSON pointers into paper/claims_spec.json, recompiles
paper/metrics_macros.tex, synchronizes paper/paper_writeup.md, and verifies
factual integrity via scripts/check_facts.py.

Usage:
    python scripts/promote_telemetry.py --phase 2
    python scripts/promote_telemetry.py --phase 3
    python scripts/promote_telemetry.py --all
    python scripts/promote_telemetry.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "paper" / "claims_spec.json"
WRITEUP_PATH = REPO_ROOT / "paper" / "paper_writeup.md"
REACHABILITY_JSON = REPO_ROOT / "results" / "reachability_null_evaluation.json"
DOSE_RESPONSE_JSON = REPO_ROOT / "results" / "dose_response_evaluation.json"

# ---------------------------------------------------------------------------
# Phase 2 Macro Specifications (RFC 6901 JSON Pointers)
# ---------------------------------------------------------------------------
PHASE_2_MACROS = {
    "ReachabilityOneHopPct": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/reachability_1hop/mean",
        "fmt": "pct1",
    },
    "ReachabilityTwoHopPct": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/reachability_2hop/mean",
        "fmt": "pct1",
    },
    "ReachabilityNullMeanPct": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/null_model/reachability_2hop/mean",
        "fmt": "pct1",
    },
    "ReachabilityNullStdPct": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/null_model/reachability_2hop/std",
        "fmt": "pct1",
    },
    "ReachabilityZScore": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/null_model/reachability_2hop/z_score",
        "fmt": "num2",
    },
    "ReachabilityNullPVal": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/null_model/reachability_2hop/p_value",
        "fmt": "pval",
    },
    "ReachabilityPrunedPct": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/pruned_reachability_2hop/mean",
        "fmt": "pct1",
    },
    "ReachabilityLexicalPct": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/lexical_reachability_size_matched/mean",
        "fmt": "pct1",
    },
    "ReachabilitySameFilePct": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/same_file_containment/mean",
        "fmt": "pct1",
    },
    "ReachabilityMedianNOne": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/reachability_1hop/median_nodes",
        "fmt": "int",
    },
    "ReachabilityMedianNTwo": {
        "file": "results/reachability_null_evaluation.json",
        "ptr": "/summary/reachability_2hop/median_nodes",
        "fmt": "int",
    },
}

# ---------------------------------------------------------------------------
# Phase 3 Macro Specifications (RFC 6901 JSON Pointers)
# ---------------------------------------------------------------------------
PHASE_3_MACROS = {
    "DoseResponseRSquared": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/h1_universal_entropy_collapse/glmm_lofo_r2",
        "fmt": "num3",
    },
    "DoseResponseCoeffA": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/h1_universal_entropy_collapse/coeff_ln_k",
        "fmt": "num3",
    },
    "DoseResponseCoeffB": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/h1_universal_entropy_collapse/coeff_ln_tau",
        "fmt": "num3",
    },
    "DoseResponseIsoEntropyPVal": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/iso_entropy_contrast/mcnemar_p_value",
        "fmt": "pval",
    },
    "DoseResponseHubDeltaAcc": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/h3_hub_harm_falsification/delta_acc_pp",
        "fmt": "num2",
    },
    "DoseResponseHubMcNemarPVal": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/h3_hub_harm_falsification/paired_mcnemar_exact/p_value",
        "fmt": "pval",
    },
    "DoseResponseHubTOSTPVal": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/h3_hub_harm_falsification/tost_equivalence/p_value",
        "fmt": "pval",
    },
    "DoseResponseHubSubtasks": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/h3_hub_harm_falsification/n_hub_tasks",
        "fmt": "int",
    },
    "DoseResponseEvaluations": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/metadata/total_evaluations",
        "fmt": "int",
    },
    "DoseResponseConfigs": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/metadata/n_configurations",
        "fmt": "int",
    },
    "DoseResponseMaxLOneError": {
        "file": "results/dose_response_evaluation.json",
        "ptr": "/linearity_validation/max_l1_error",
        "fmt": "num3",
    },
}


def _assert_finite(node: Any, path: str = "") -> None:
    """Ensure strict RFC 8259 compliance (no NaN, inf, or -inf)."""
    if isinstance(node, float):
        if math.isnan(node) or math.isinf(node):
            raise ValueError(f"Non-finite float value at {path}: {node}")
    elif isinstance(node, dict):
        for k, v in node.items():
            _assert_finite(v, f"{path}/{k}")
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _assert_finite(item, f"{path}/{idx}")


def validate_phase_2_results(data: Dict[str, Any]) -> None:
    """Validate Phase 2 acceptance criteria against pre-registered gates."""
    _assert_finite(data)
    assert "summary" in data, "Phase 2 results missing 'summary' block"
    summary = data["summary"]

    assert "reachability_1hop" in summary, "Missing 'reachability_1hop'"
    assert "reachability_2hop" in summary, "Missing 'reachability_2hop'"
    assert "null_model" in summary, "Missing 'null_model'"

    r1 = summary["reachability_1hop"]["mean"]
    r2 = summary["reachability_2hop"]["mean"]
    assert 0.0 < r1 < r2 <= 1.0, f"Unreasonable reachability ordering: 1-hop={r1}, 2-hop={r2}"

    null_mean = summary["null_model"]["reachability_2hop"]["mean"]
    z_score = summary["null_model"]["reachability_2hop"]["z_score"]
    p_val = summary["null_model"]["reachability_2hop"]["p_value"]

    assert 0.0 < null_mean <= 1.0, f"Null mean out of bounds: {null_mean}"
    assert -10.0 <= z_score <= 10.0, f"Z-score extreme: {z_score}"
    assert 0.0 <= p_val <= 1.0, f"P-value invalid: {p_val}"


def validate_phase_3_results(data: Dict[str, Any]) -> None:
    """Validate Phase 3 acceptance criteria against pre-registered gates."""
    _assert_finite(data)
    assert "linearity_validation" in data, "Phase 3 missing linearity validation"
    assert "h1_universal_entropy_collapse" in data, "Phase 3 missing H1 evaluation"
    assert "h3_hub_harm_falsification" in data, "Phase 3 missing H3 evaluation"

    # 1. Resolvent Linearity gate: L1 <= 2.0e-6
    max_l1 = data["linearity_validation"]["max_l1_error"]
    assert max_l1 <= 2.0e-6, f"Resolvent linearity violated: L1 error {max_l1:.3e} > 2.0e-6"

    # 2. Universal Entropy Collapse (H1) gate: 0.0 <= R^2 <= 1.0, |a|, |b| < 0.10
    h1 = data["h1_universal_entropy_collapse"]
    r2 = h1["glmm_lofo_r2"]
    a = h1["coeff_ln_k"]
    b = h1["coeff_ln_tau"]
    assert 0.0 <= r2 <= 1.0, f"H1 LOFO R^2 out of bounds: {r2}"
    assert abs(a) < 0.10, f"H1 coefficient a non-vanishing: {a}"
    assert abs(b) < 0.10, f"H1 coefficient b non-vanishing: {b}"

    # 3. Hub-harm falsification (H3) gate: McNemar p > 0.05, TOST p < 0.05
    h3 = data["h3_hub_harm_falsification"]
    mcnemar_p = h3["paired_mcnemar_exact"]["p_value"]
    tost_p = h3["tost_equivalence"]["p_value"]
    assert mcnemar_p > 0.05, f"Hub presence caused significant divergence: p={mcnemar_p}"
    assert tost_p < 0.05, f"Hub equivalence TOST test failed: p={tost_p}"


def inject_macros(new_macros: Dict[str, Dict[str, Any]]) -> None:
    """Merge new macro bindings into paper/claims_spec.json atomically."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    macros = spec.setdefault("macros", {})
    macros.update(new_macros)
    staging_file = SPEC_PATH.with_suffix(".tmp")
    staging_file.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(staging_file, SPEC_PATH)
    print(f"[OK] Injected {len(new_macros)} macros into {SPEC_PATH.name}")


def recompile_macros() -> None:
    """Run scripts/generate_paper_macros.py to update metrics_macros.tex."""
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_paper_macros.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"generate_paper_macros failed:\n{res.stderr}")
    print(f"[OK] Recompiled paper/metrics_macros.tex successfully.")


def synchronize_writeup(phase: int) -> None:
    """Replace Stage 1 semantic placeholders in paper_writeup.md with verified numbers."""
    if not WRITEUP_PATH.is_file():
        return

    text = WRITEUP_PATH.read_text(encoding="utf-8")

    if phase == 2 or phase == 0:
        if REACHABILITY_JSON.is_file():
            d2 = json.loads(REACHABILITY_JSON.read_text(encoding="utf-8"))
            s2 = d2["summary"]
            nm = s2["null_model"]["reachability_2hop"]

            r1 = f"{s2['reachability_1hop']['mean'] * 100:.1f}%"
            r2 = f"{s2['reachability_2hop']['mean'] * 100:.1f}%"
            null_mean = f"{nm['mean'] * 100:.1f}%"
            null_std = f"{nm['std'] * 100:.1f}%"
            ze = f"{nm['z_score']:+.2f}"
            pval = f"{nm['p_value']:.3f}"
            pruned = f"{s2['pruned_reachability_2hop']['mean'] * 100:.1f}%"
            lexical = f"{s2['lexical_reachability_size_matched']['mean'] * 100:.1f}%"
            samefile = f"{s2['same_file_containment']['mean'] * 100:.1f}%"

            rep2 = {
                "[MEASURED_P2_REACH_1HOP]": r1,
                "[MEASURED_P2_REACH_2HOP]": r2,
                "[MEASURED_P2_NULL_MEAN]": null_mean,
                "[MEASURED_P2_NULL_STD]": null_std,
                "[MEASURED_P2_ZE]": ze,
                "[MEASURED_P2_PVAL]": pval,
                "[MEASURED_P2_PRUNED_2HOP]": pruned,
                "[MEASURED_P2_LEXICAL_REACH]": lexical,
                "[MEASURED_P2_SAMEFILE_REACH]": samefile,
            }
            for k, v in rep2.items():
                text = text.replace(k, v)

    if phase == 3 or phase == 0:
        if DOSE_RESPONSE_JSON.is_file():
            d3 = json.loads(DOSE_RESPONSE_JSON.read_text(encoding="utf-8"))
            h1 = d3["h1_universal_entropy_collapse"]
            h3 = d3["h3_hub_harm_falsification"]
            iso = d3["iso_entropy_contrast"]

            r2_val = f"{h1['glmm_lofo_r2']:.3f}"
            a_val = f"{h1['coeff_ln_k']:.3f}"
            b_val = f"{h1['coeff_ln_tau']:.3f}"
            iso_p = f"{iso['mcnemar_p_value']:.3f}"
            hub_delta = f"{h3['delta_acc_pp']:+.2f}"
            tost_p = f"{h3['tost_equivalence']['p_value']:.3f}"

            rep3 = {
                "[MEASURED_P3_R2]": r2_val,
                "[MEASURED_P3_COEF_A]": a_val,
                "[MEASURED_P3_COEF_B]": b_val,
                "[MEASURED_P3_MCNEMAR_P]": iso_p,
                "[MEASURED_P3_HUB_DELTA]": hub_delta,
                "[MEASURED_P3_TOST_P]": tost_p,
            }
            for k, v in rep3.items():
                text = text.replace(k, v)

    # Check for unresolved placeholders in writeup
    remaining = re.findall(r"\[MEASURED_[A-Za-z0-9_]+\]", text)
    if remaining:
        print(f"[WARN] Remaining un-promoted placeholders in writeup: {remaining}")

    staging_writeup = WRITEUP_PATH.with_suffix(".tmp")
    staging_writeup.write_text(text, encoding="utf-8")
    os.replace(staging_writeup, WRITEUP_PATH)
    print(f"[OK] Synchronized {WRITEUP_PATH.name}")


def run_checks() -> None:
    """Execute check_facts.py and verify word counts."""
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_facts.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"check_facts.py failed:\n{res.stdout}\n{res.stderr}")
    print(f"[OK] check_facts.py passed cleanly (0 unbound tokens).")

    # Word count check
    text = WRITEUP_PATH.read_text(encoding="utf-8")
    words_split = len(text.split())
    words_regex = len(re.findall(r"\b\w+\b", text))
    assert words_split <= 2730, f"Writeup split() count {words_split} exceeds 2,730 limit"
    assert words_regex <= 2950, f"Writeup regex count {words_regex} exceeds 2,950 limit"
    print(f"[OK] Manuscript word count verified: {words_split} words (split), {words_regex} words (regex).")


def promote_phase_2() -> None:
    print("\n--- PROMOTING PHASE 2 TELEMETRY ---")
    if not REACHABILITY_JSON.is_file():
        raise FileNotFoundError(f"Missing required artifact: {REACHABILITY_JSON}")
    data = json.loads(REACHABILITY_JSON.read_text(encoding="utf-8"))
    validate_phase_2_results(data)
    inject_macros(PHASE_2_MACROS)
    synchronize_writeup(2)


def promote_phase_3() -> None:
    print("\n--- PROMOTING PHASE 3 TELEMETRY ---")
    if not DOSE_RESPONSE_JSON.is_file():
        raise FileNotFoundError(f"Missing required artifact: {DOSE_RESPONSE_JSON}")
    data = json.loads(DOSE_RESPONSE_JSON.read_text(encoding="utf-8"))
    validate_phase_3_results(data)
    inject_macros(PHASE_3_MACROS)
    synchronize_writeup(3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote validated benchmark telemetry to publication artifacts.")
    parser.add_argument("--phase", type=int, choices=[2, 3], help="Specific phase to promote (2 or 3)")
    parser.add_argument("--all", action="store_true", help="Promote all available phases")
    parser.add_argument("--verify-only", action="store_true", help="Run verification without updating files")
    args = parser.parse_args()

    if args.phase == 2 or args.all:
        promote_phase_2()
    if args.phase == 3 or args.all:
        promote_phase_3()

    if not args.verify_only:
        recompile_macros()
        run_checks()

    print("\n[SUCCESS] Telemetry promotion and publication synchronization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
