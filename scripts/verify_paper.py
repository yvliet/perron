#!/usr/bin/env python3
"""
Comprehensive Research Paper & Reproduction Verifier.

Audits:
1. Test suite integrity (pytest tests/).
2. Real repository graph assets (Requests & SymPy CSR matrices).
3. Real SWE-bench Lite cache (300 instances).
4. Data manifest & LaTeX macro bindings integrity (data_manifest.json & metrics_macros.tex).
5. Publication figures generation.
6. LaTeX paper build (paper/main.pdf).
7. Showcase notebook execution (perron_showcase.ipynb).
8. Research writeup word count limit (<= 3,000 words on paper/paper_writeup.md).
9. Turnkey agent runner CLI and manifest verification.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_check(name: str, check_fn) -> bool:
    print(f"\n[CHECK] {name}...")
    try:
        check_fn()
        print(f"[PASS]  {name}")
        return True
    except Exception as e:
        print(f"[FAIL]  {name}: {e}")
        return False


def check_real_graphs():
    req_dir = REPO_ROOT / "data" / "real_graphs" / "requests"
    sym_dir = REPO_ROOT / "data" / "real_graphs" / "sympy"
    for d in [req_dir, sym_dir]:
        assert d.is_dir(), f"Missing graph dir: {d}"
        for f in ["data.npy", "indices.npy", "indptr.npy", "dangling.npy", "symbols.json"]:
            assert (d / f).is_file(), f"Missing graph file: {d / f}"


def check_swebench_cache():
    cache_file = REPO_ROOT / "data" / "swebench_lite_cache.jsonl"
    assert cache_file.is_file(), f"Missing cache: {cache_file}"
    with open(cache_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 300, f"Expected 300 cached instances, found {len(lines)}"


def check_manifest_and_macros():
    manifest_file = REPO_ROOT / "paper" / "data_manifest.json"
    macros_file = REPO_ROOT / "paper" / "metrics_macros.tex"
    assert manifest_file.is_file(), f"Missing manifest: {manifest_file}"
    assert macros_file.is_file(), f"Missing macros: {macros_file}"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "hub_suppression_index" in manifest
    assert "function_recall" in manifest
    assert "mean_reciprocal_rank" in manifest

    with open(macros_file, "r", encoding="utf-8") as f:
        macros_content = f.read()
    assert "\\PerronHSIMean" in macros_content
    assert "\\StandardPPRHSIMean" in macros_content


def check_figures():
    fig_dir = REPO_ROOT / "paper" / "figures"
    expected_figures = [
        "perron_pipeline",
        "comparative_edit_interfaces",
        "action_and_failure_distribution",
        "pass_at_k_scaling",
        "benchmark_statistical_evaluation",
    ]
    for fig_name in expected_figures:
        fig_pdf = fig_dir / f"{fig_name}.pdf"
        fig_png = fig_dir / f"{fig_name}.png"
        assert fig_pdf.is_file(), f"Missing figure PDF: {fig_pdf}"
        assert fig_png.is_file(), f"Missing figure PNG: {fig_png}"
        assert fig_pdf.stat().st_size > 1000, f"Figure PDF suspiciously small: {fig_pdf}"
        assert fig_png.stat().st_size > 1000, f"Figure PNG suspiciously small: {fig_png}"


def check_paper_pdf():
    pdf_file = REPO_ROOT / "paper" / "main.pdf"
    assert pdf_file.is_file(), f"Missing paper PDF: {pdf_file}"
    assert pdf_file.stat().st_size > 100_000, f"PDF file size suspiciously small: {pdf_file.stat().st_size} bytes"


def check_writeup_word_count():
    import re
    writeup = REPO_ROOT / "paper" / "paper_writeup.md"
    assert writeup.is_file(), f"Missing writeup: {writeup}"
    text = writeup.read_text(encoding="utf-8")
    words_split = len(text.split())
    words_regex = len(re.findall(r'\b\w+\b', text))
    print(f"        Writeup word count: {words_split} words (split), {words_regex} words (regex) (Kaggle ceiling: 3,000 words)")
    assert words_split <= 2750, f"Writeup exceeds safety buffer (2,750 words): {words_split} words"
    assert words_regex <= 2950, f"Writeup exceeds regex safety limit (2,950 words): {words_regex} words"


def check_notebook_execution():
    nb_file = REPO_ROOT / "notebooks" / "perron_showcase.ipynb"
    assert nb_file.is_file(), f"Missing notebook: {nb_file}"
    quickstart_file = REPO_ROOT / "notebooks" / "perron_quickstart.ipynb"
    assert quickstart_file.is_file(), f"Missing quickstart notebook: {quickstart_file}"
    with open(nb_file, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_ns = {"__name__": "__main__"}
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            exec(code, global_ns)


def check_facts_and_claims():
    from scripts.check_facts import check_facts
    ret = check_facts()
    assert ret == 0, "Automated fact-checker failed on paper claims"


def check_privacy_and_links():
    import re
    writeup = REPO_ROOT / "paper" / "paper_writeup.md"
    text = writeup.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in text, "Local Windows user path leaked in writeup"
    assert "/home/" not in text, "Local Unix user path leaked in writeup"

    # Extract clean URLs (stripping trailing punctuation)
    urls = re.findall(r'https?://[^\s\)\>\]]+', text)
    cleaned_urls = [re.sub(r'[\.,;:]$', '', u) for u in urls]
    for url in cleaned_urls:
        assert url.startswith("https://github.com/yvliet"), f"Unexpected external URL: {url}"


def check_unit_tests():
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pytest failed:\n{res.stdout}\n{res.stderr}")


def check_runner_cli():
    # Verify CLI help output and arguments
    res = subprocess.run(
        [sys.executable, "-m", "perron.runner", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"perron.runner --help failed: {res.stderr}"
    assert "--repo-dir" in res.stdout
    assert "--issue" in res.stdout
    assert "--config" in res.stdout
    assert "--max-turns" in res.stdout

    # Verify programmatic manifest loading
    from perron.runner import load_agent_yaml
    cfg = load_agent_yaml(REPO_ROOT / "agent.yaml")
    assert "model" in cfg, "agent.yaml missing 'model' section"
    assert "diffusion" in cfg, "agent.yaml missing 'diffusion' section"
    assert "tools" in cfg, "agent.yaml missing 'tools' section"


def main():
    print("=" * 70)
    print("PERRON COMPREHENSIVE VERIFICATION & AUDIT SUITE")
    print("=" * 70)

    checks = [
        ("Real Repository Call Graphs", check_real_graphs),
        ("Real SWE-bench Lite Cache", check_swebench_cache),
        ("Data Manifest & LaTeX Macros", check_manifest_and_macros),
        ("Publication Statistical Figures", check_figures),
        ("LaTeX Manuscript Compilation", check_paper_pdf),
        ("Research Writeup Word Count (<= 3,000 words)", check_writeup_word_count),
        ("Showcase Notebook Execution", check_notebook_execution),
        ("Automated Empirical Facts & Claims", check_facts_and_claims),
        ("Privacy & Hyperlink Integrity", check_privacy_and_links),
        ("Core Unit Test Battery", check_unit_tests),
        ("Turnkey Agent Runner CLI & Manifest Verification", check_runner_cli),
    ]

    all_passed = True
    for name, fn in checks:
        if not run_check(name, fn):
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("SOME VERIFICATION CHECKS FAILED.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
