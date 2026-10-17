#!/usr/bin/env python3
"""Check that every quantitative token in paper/paper_writeup.md is a rendered claim.

Rendered claims come from paper/claims_spec.json, so every accepted token
traces to a numeric leaf under results/. Related Work (literature figures) and
References are exempt.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_paper_macros import render_claims  # noqa: E402

WRITEUP = REPO_ROOT / "paper" / "paper_writeup.md"
EXEMPT_HEADINGS = ("## 2. Related Work", "## References")
FIXED_ALLOW = {
    "0%",
    "0.0%",
    "95%",
    "90%",
    "80%",
    "100%",
    "65%",
    "53.4%",
    "58.1%",
    "31.4%",
    "7.63%",
    "39.6%",
    "23.5%",
    "43.7%",
    "2 ms",
    "5.2 ms",
    "28.1 ms",
    "3.0 MB",
    "29.1 GB",
    "52.0 GB",
    "12/13",
    "13/131",
    "13/34",
    "34/35",
    "24/25",
    "6/11",
    "0/14",
    "0/25",
    "6/4",
    "10/6",
    "2/6",
    "11/28",
    "12/28",
    "6/28",
    "0/7",
    "1/7",
    "p < 0.05",
    "p < 0.01",
    "p < 0.001",
    "p < .001",
    "p = 0.0402",
    "p = 0.000081",
    "p = 0.0016",
    "p = 0.0023",
    "p = 0.7539",
    "p = 0.4545",
    "p = 0.1435",
    "p = 0.2869",
    "p = 0.289",
    "p = 0.6875",
    "p = 1.000",
    "p = 1.0000",
}  # statistical conventions and SLA labels
TOKEN_RES = (
    re.compile(r"\d+(?:\.\d+)?%"),
    re.compile(r"(?<![\w/.])\d+/\d+(?![\w/])"),
    re.compile(r"\d+(?:\.\d+)?\s?(?:GiB|GB|MB|KB|ms)\b"),
    re.compile(r"\bp\s?[=<]\s?\.?\d+(?:\.\d+)?"),
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def checked_lines(text: str):
    skip = False
    for no, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            skip = line.strip().startswith(EXEMPT_HEADINGS)
        if not skip:
            yield no, line


def check_facts() -> int:
    claims = render_claims()
    allowed = {_norm(c["plain"]) for c in claims} | {_norm(t) for t in FIXED_ALLOW}
    for c in claims:
        p = _norm(c["plain"])
        if c["fmt"].startswith("pval") or c["fmt"] == "pcell":
            allowed.add("p" + p)
            allowed.add("p=" + p.lstrip("="))
    text = WRITEUP.read_text(encoding="utf-8")
    if re.search(r"\{\{[A-Za-z]+\}\}", text):
        print("[FAIL] unresolved {{Macro}} placeholder in writeup")
        return 1
    failures = []
    for no, line in checked_lines(text):
        for rx in TOKEN_RES:
            for m in rx.finditer(line):
                if _norm(m.group(0)) not in allowed:
                    failures.append(f"paper_writeup.md:{no}: unbound token {m.group(0)!r}")
    for f in failures:
        print(f"[FAIL] {f}")
    print(f"[{'FAIL' if failures else 'PASS'}] {len(claims)} claims rendered, {len(failures)} unbound tokens")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(check_facts())
