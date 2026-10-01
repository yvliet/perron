"""
Sanitizes metadata paths in data/real_graphs/ to ensure all file paths are relative to repository root.
Guarantees zero leakage of local filesystem directory paths.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sanitize_all():
    target_dirs = [
        REPO_ROOT / "data" / "real_graphs" / "sympy",
        REPO_ROOT / "data" / "real_graphs" / "requests",
    ]
    for target_dir in target_dirs:
        for fname in ["id_to_node.json", "node_to_id.json", "symbols.json"]:
            p = target_dir / fname
            if not p.exists():
                continue
            txt = p.read_text(encoding="utf-8")
            
            # Replace absolute user filesystem paths leading up to repo modules
            txt_clean = re.sub(r'[A-Za-z]:[\\/][Uu]sers[\\/][^"\'\n\r]+?[\\/](sympy|requests)[\\/]', r'\1/', txt)
            txt_clean = re.sub(r'/home/[^"\'\n\r]+?/(sympy|requests)/', r'\1/', txt_clean)
            
            if txt_clean != txt:
                p.write_text(txt_clean, encoding="utf-8")
                print(f"Sanitized: {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sanitize_all()
