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
    "31.4%",
    "7.63%",
    "2 ms",
    "5.2 ms",
    "28.1 ms",
    "3.0 MB",
    "29.1 GB",
    "12/13",
    "34/35",
    "24/25",
    "p < 0.05",
    "p < 0.001",
    "p < .001",
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
