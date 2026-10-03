"""
Compile real repository graphs for Requests and SymPy into zero-copy mmap CSR matrices.
"""

import json
from pathlib import Path
import sympy
import requests

from perron.graph import extract_repository_graph
from perron.matrix import build_static_transition_matrix, save_mmap_csr

REPO_ROOT = Path(__file__).resolve().parent.parent

def compile_real_graphs():
    out_base = REPO_ROOT / "data" / "real_graphs"
    out_base.mkdir(parents=True, exist_ok=True)

    # 1. Requests
    req_dir = out_base / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    pkg_req = Path(requests.__file__).parent
    print(f"Extracting Requests from {pkg_req}...")
    r_syms, r_calls, r_callers, r_c2id, r_id2c, r_inherit, r_import = extract_repository_graph(
        pkg_req, return_multiplex=True
    )
    r_t, r_dang = build_static_transition_matrix(
        len(r_syms), r_calls, r_callers,
        inherit_edges=r_inherit, import_edges=r_import
    )
    save_mmap_csr(req_dir, r_t, r_dang, r_c2id, r_id2c)
    with open(req_dir / "symbols.json", "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in r_syms], f, indent=2)
    print(f"Compiled Requests: {len(r_syms)} symbols, {r_t.nnz} edges into {req_dir}")

    # 2. SymPy
    sym_dir = out_base / "sympy"
    sym_dir.mkdir(parents=True, exist_ok=True)
    pkg_sym = Path(sympy.__file__).parent
    subpackages = ["printing", "core", "matrices", "functions", "polys", "utilities", "simplify", "solvers"]
    
    all_syms, all_calls, all_callers = [], [], []
    all_inherit, all_imports = [], []
    curr_offset = 0
    all_c2id = {}
    all_id2c = {}

    print(f"Extracting SymPy subpackages from {pkg_sym}...")
    for sp in subpackages:
        sub_p = pkg_sym / sp
        if sub_p.is_dir():
            s, c, cr, c2i, i2c, inh, imp = extract_repository_graph(
                sub_p, max_files=25, return_multiplex=True
            )
            for node in s:
                node.node_id += curr_offset
                all_syms.append(node)
                all_c2id[node.canonical_key] = node.node_id
                all_id2c[node.node_id] = node.canonical_key
            for u, v in c:
                all_calls.append((u + curr_offset, v + curr_offset))
            for u, v in cr:
                all_callers.append((u + curr_offset, v + curr_offset))
            for u, v in inh:
                all_inherit.append((u + curr_offset, v + curr_offset))
            for u, v in imp:
                all_imports.append((u + curr_offset, v + curr_offset))
            curr_offset = len(all_syms)
            print(f"  -> {sp}: {len(s)} symbols, {len(c)+len(cr)+len(inh)+len(imp)} edges")

    s_t, s_dang = build_static_transition_matrix(
        len(all_syms), all_calls, all_callers,
        inherit_edges=all_inherit, import_edges=all_imports
    )
    save_mmap_csr(sym_dir, s_t, s_dang, all_c2id, all_id2c)
    with open(sym_dir / "symbols.json", "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in all_syms], f, indent=2)
    print(f"Compiled SymPy: {len(all_syms)} symbols, {s_t.nnz} edges into {sym_dir}")

if __name__ == "__main__":
    compile_real_graphs()
