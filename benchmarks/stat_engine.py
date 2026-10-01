#!/usr/bin/env python3
"""
stat_engine.py - Statistical analysis engine for Perron evaluation benchmarks.
Calculates Welch's t-test, Mann-Whitney U, Wilcoxon signed-rank test,
Cliff's delta, Cohen's d, and bootstrap confidence intervals.
Outputs data manifests and LaTeX macro bindings.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats


def bootstrap_ci(
    data: Any,
    stat_func=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval."""
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0, 0.0
    n = len(arr)
    rng = np.random.default_rng(42)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = stat_func(sample)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
    return lower, upper


def compute_cliffs_delta(x: Any, y: Any) -> float:
    """Compute non-parametric Cliff's delta effect size."""
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    arr_x = arr_x[~np.isnan(arr_x)]
    arr_y = arr_y[~np.isnan(arr_y)]
    n1, n2 = len(arr_x), len(arr_y)
    if n1 == 0 or n2 == 0:
        return 0.0
    diff = arr_x[:, None] - arr_y[None, :]
    delta = (np.sum(diff > 0) - np.sum(diff < 0)) / (n1 * n2)
    return float(delta)


def compute_cohens_d(x: Any, y: Any) -> float:
    """Compute Cohen's d with pooled standard deviation."""
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    arr_x = arr_x[~np.isnan(arr_x)]
    arr_y = arr_y[~np.isnan(arr_y)]
    n1, n2 = len(arr_x), len(arr_y)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(arr_x, ddof=1), np.var(arr_y, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(arr_x) - np.mean(arr_y)) / pooled_sd)


def compute_iqm(scores: Any) -> float:
    """Compute Interquartile Mean (IQM) across evaluation tasks."""
    arr = np.asarray(scores, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    trimmed = arr[(arr >= q25) & (arr <= q75)]
    return float(np.mean(trimmed) if len(trimmed) > 0 else np.mean(arr))


def analyze_two_groups(
    group_a: Any,
    group_b: Any,
    label_a: str = "Group A",
    label_b: str = "Group B",
    paired: bool = False,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """
    Robust two-group statistical evaluation pipeline:
    Calculates Welch's t-test, Mann-Whitney U or Wilcoxon signed-rank test,
    Cliff's delta, Cohen's d, and 95% bootstrap confidence intervals.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)

    if n_a < 3 or n_b < 3:
        return {"error": "Sample size too small for statistical inference (N < 3)."}

    mean_a, std_a = float(np.mean(a)), float(np.std(a, ddof=1))
    mean_b, std_b = float(np.mean(b)), float(np.std(b, ddof=1))
    median_a, iqr_a = float(np.median(a)), float(stats.iqr(a))
    median_b, iqr_b = float(np.median(b)), float(stats.iqr(b))
    iqm_a, iqm_b = compute_iqm(a), compute_iqm(b)

    # Normality diagnostics
    _, p_norm_a = stats.shapiro(a) if n_a < 50 else stats.normaltest(a)
    _, p_norm_b = stats.shapiro(b) if n_b < 50 else stats.normaltest(b)
    _, p_levene = stats.levene(a, b)

    diff_mean = mean_b - mean_a
    rng = np.random.default_rng(42)
    boot_diffs = np.empty(2000)
    for i in range(2000):
        samp_a = rng.choice(a, size=n_a, replace=True)
        samp_b = rng.choice(b, size=n_b, replace=True)
        boot_diffs[i] = np.mean(samp_b) - np.mean(samp_a)
    ci_diff_low = float(np.percentile(boot_diffs, 2.5))
    ci_diff_high = float(np.percentile(boot_diffs, 97.5))

    if paired:
        if n_a != n_b:
            return {"error": "Paired comparison requires identical sample sizes."}
        diff = b - a
        res_paired = stats.ttest_rel(b, a)
        try:
            res_wilcoxon = stats.wilcoxon(b, a)
            w_stat = float(res_wilcoxon.statistic)
            w_pval = float(res_wilcoxon.pvalue)
        except Exception:
            w_stat = 0.0
            w_pval = 1.0

        d_val = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
        delta_val = compute_cliffs_delta(b, a)

        report: Dict[str, Any] = {
            "test_type": "Paired Samples Evaluation",
            "primary_test": "Paired Student's t-test",
            "statistic": float(res_paired.statistic),
            "df": n_a - 1,
            "p_value": float(res_paired.pvalue),
            "wilcoxon_w": w_stat,
            "wilcoxon_p": w_pval,
            "mean_diff": float(np.mean(diff)),
            "ci_diff_95": (ci_diff_low, ci_diff_high),
            "effect_size_cohen_d": d_val,
            "effect_size_cliffs_delta": delta_val,
        }
    else:
        res_welch = stats.ttest_ind(b, a, equal_var=False)
        v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
        df_welch = (v1 / n_a + v2 / n_b) ** 2 / (
            (v1 / n_a) ** 2 / (n_a - 1) + (v2 / n_b) ** 2 / (n_b - 1)
        )
        res_mwu = stats.mannwhitneyu(b, a, alternative="two-sided")
        d_val = compute_cohens_d(b, a)
        delta_val = compute_cliffs_delta(b, a)

        report = {
            "test_type": "Two-Sample Continuous Comparison",
            "primary_test": "Welch's t-test",
            "statistic_t": float(res_welch.statistic),
            "df_welch": round(float(df_welch), 1),
            "p_value_welch": float(res_welch.pvalue),
            "secondary_test": "Mann-Whitney U",
            "statistic_u": float(res_mwu.statistic),
            "p_value_mwu": float(res_mwu.pvalue),
            "mean_diff": diff_mean,
            "ci_diff_95": (ci_diff_low, ci_diff_high),
            "effect_size_cohen_d": d_val,
            "effect_size_cliffs_delta": delta_val,
        }

    report.update({
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": mean_a,
        "std_a": std_a,
        "median_a": median_a,
        "iqr_a": iqr_a,
        "iqm_a": iqm_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "median_b": median_b,
        "iqr_b": iqr_b,
        "iqm_b": iqm_b,
        "diagnostics": {
            "p_norm_a": float(p_norm_a),
            "p_norm_b": float(p_norm_b),
            "p_levene": float(p_levene),
        },
    })

    p_val = report.get("p_value_welch", report.get("p_value"))
    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < .001"
    stat_val = report.get("statistic_t", report.get("statistic"))
    df_val = report.get("df_welch", report.get("df"))

    report["academic_prose"] = (
        f"A statistical comparison demonstrated a "
        f"{'statistically significant' if p_val < 0.05 else 'non-significant'} difference between "
        f"{label_b} (M = {mean_b:.2f}, SD = {std_b:.2f}, IQM = {iqm_b:.2f}) and "
        f"{label_a} (M = {mean_a:.2f}, SD = {std_a:.2f}, IQM = {iqm_a:.2f}), "
        f"t({df_val}) = {stat_val:.2f}, {p_str}, "
        f"mean difference = {diff_mean:.2f} (95% CI [{ci_diff_low:.2f}, {ci_diff_high:.2f}]), "
        f"Cohen's d = {d_val:.2f}, Cliff's delta = {delta_val:.2f}."
    )

    export_data_manifest(report, label_a, label_b, output_dir)
    return report


def export_data_manifest(
    report: Dict[str, Any],
    label_a: str,
    label_b: str,
    output_dir: str,
) -> None:
    """Exports structured data manifest and LaTeX macro bindings."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "data_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    clean_a = "".join(filter(str.isalnum, label_a))
    clean_b = "".join(filter(str.isalnum, label_b))

    p_val = report.get("p_value_welch", report.get("p_value", 0.0))
    p_text = f"{p_val:.3f}" if p_val >= 0.001 else "< .001"
    t_val = report.get("statistic_t", report.get("statistic", 0.0))
    df_val = report.get("df_welch", report.get("df", 0.0))

    macros_content = f"""% Auto-generated by stat_engine.py. DO NOT EDIT MANUALLY.
% Immutable Data Manifest Bindings
\\newcommand{{\\{clean_a}Mean}}{{{report['mean_a']:.2f}}}
\\newcommand{{\\{clean_a}Std}}{{{report['std_a']:.2f}}}
\\newcommand{{\\{clean_a}IQM}}{{{report['iqm_a']:.2f}}}
\\newcommand{{\\{clean_b}Mean}}{{{report['mean_b']:.2f}}}
\\newcommand{{\\{clean_b}Std}}{{{report['std_b']:.2f}}}
\\newcommand{{\\{clean_b}IQM}}{{{report['iqm_b']:.2f}}}
\\newcommand{{\\MeanDiff}}{{{report['mean_diff']:.2f}}}
\\newcommand{{\\MeanDiffCILow}}{{{report['ci_diff_95'][0]:.2f}}}
\\newcommand{{\\MeanDiffCIHigh}}{{{report['ci_diff_95'][1]:.2f}}}
\\newcommand{{\\StatTValue}}{{{t_val:.2f}}}
\\newcommand{{\\StatDF}}{{{df_val}}}
\\newcommand{{\\StatPValue}}{{{p_text}}}
\\newcommand{{\\CohensD}}{{{report['effect_size_cohen_d']:.2f}}}
\\newcommand{{\\CliffsDelta}}{{{report['effect_size_cliffs_delta']:.2f}}}
"""
    macros_path = target_dir / "metrics_macros.tex"
    with open(macros_path, "w", encoding="utf-8") as f:
        f.write(macros_content)
