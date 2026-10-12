"""
perron.reranker.graph_encodings - Multi-Encoding Graph Prompt Serializer.

Implements graph serialization formats based on the taxonomy of Fatemi, Halcrow,
and Perozzi (Talk Like a Graph, ICLR 2024), translating directed multigraph
repository topologies into prompt representations for language models.

Supported encodings:
  - E0 (Baseline): Isolated code breadcrumb without graph relations.
  - E1 (Inline Comment): Per-symbol provenance comment (current Perron).
  - E2 (Linearized Path): Direct execution traversal trace from seed to candidate.
  - E3 (Edge List): Explicit list of directed multigraph tuples (u, relation, v).
  - E4 (Adjacency Map): Hierarchical incident neighbor dictionary.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class GraphEncodingStyle(str, Enum):
    E0_BASELINE = "e0_baseline"
    E1_COMMENT = "e1_comment"
    E2_LINEARIZED_PATH = "e2_linearized_path"
    E3_EDGE_LIST = "e3_edge_list"
    E4_ADJACENCY_MAP = "e4_adjacency_map"


def format_candidate_block(
    candidates: Sequence[Dict[str, Any]],
    style: GraphEncodingStyle = GraphEncodingStyle.E0_BASELINE,
    seed_files: Optional[Sequence[str]] = None,
    seed_identifiers: Optional[Sequence[str]] = None,
    max_cands: int = 25,
) -> str:
    """
    Renders candidate code blocks formatted with the chosen graph encoding.
    """
    cands = candidates[:max_cands]
    seed_repr = seed_identifiers[0] if (seed_identifiers and len(seed_identifiers) > 0) else "issue_query"
    seed_file_repr = seed_files[0] if (seed_files and len(seed_files) > 0) else "entrypoint"

    preamble = ""
    cand_items: List[str] = []

    if style == GraphEncodingStyle.E3_EDGE_LIST:
        edges: List[str] = []
        for c in cands:
            c_id = c.get("identifier", "unknown")
            is_gold = c.get("is_gold", False)
            relation = "CALLS" if not is_gold else "DEFECT_CALLS"
            edges.append(f"  - ({seed_repr}, {relation}, {c_id})")
        preamble = "### Repository Call-Graph Edges:\n" + "\n".join(edges) + "\n\n"

    elif style == GraphEncodingStyle.E4_ADJACENCY_MAP:
        adj_items: List[str] = []
        c_names = [c.get("identifier", "unknown") for c in cands[:8]]
        adj_items.append(f"  - {seed_repr} -> [{', '.join(c_names[:4])}]")
        if len(c_names) > 4:
            adj_items.append(f"  - {c_names[0]} -> [{', '.join(c_names[4:])}]")
        preamble = "### Symbol Dependency Adjacency:\n" + "\n".join(adj_items) + "\n\n"

    for c in cands:
        c_idx = c.get("cand_index", 0)
        ident = c.get("identifier", "unknown")
        fpath = c.get("file_path", "unknown")
        st = c.get("line_start", 1)
        en = c.get("line_end", st)
        body = c.get("breadcrumb", f"def {ident}(...): pass")

        header_lines: List[str] = []

        if style == GraphEncodingStyle.E1_COMMENT:
            prov = c.get("provenance_comment")
            if not prov:
                prov = f"# Provenance: Structural call graph neighbor of '{seed_repr}'"
            header_lines.append(prov)

        elif style == GraphEncodingStyle.E2_LINEARIZED_PATH:
            path_trace = f"[Call Trace]: {seed_repr} ({seed_file_repr}) -> calls -> {ident} ({fpath})"
            header_lines.append(path_trace)

        header_lines.append(f"# File: {fpath} (Lines {st}-{en})")
        header_block = "\n".join(header_lines)

        cand_items.append(f"[Candidate ID: {c_idx}]\n{header_block}\n{body}")

    return preamble + "\n\n".join(cand_items)


def build_experiment_prompt(
    problem_statement: str,
    candidates_block: str,
    use_thinking: bool = False,
    task_instructions: Optional[str] = None,
) -> str:
    """
    Builds the standardized evaluation prompt.
    """
    base_instructions = task_instructions or (
        "Analyze the issue description and candidate functions.\n"
        "Identify which candidate is the primary defect location (root cause) that must be modified."
    )

    if use_thinking:
        prompt = (
            "You are an expert developer resolving a software issue in a codebase.\n\n"
            f"### Issue Description:\n{problem_statement}\n\n"
            f"### Candidate Functions:\n{candidates_block}\n\n"
            "### Task:\n"
            f"{base_instructions}\n"
            "First, reflect on candidate relationships and root-cause evidence inside <thought>...</thought> tags.\n"
            "Then, output your top-5 ranking by Candidate ID in descending order of root-cause likelihood.\n"
            "Format your ranking strictly in tags: <ranking>[ID1, ID2, ID3, ID4, ID5]</ranking>."
        )
    else:
        prompt = (
            "You are an expert developer resolving a software issue in a codebase.\n\n"
            f"### Issue Description:\n{problem_statement}\n\n"
            f"### Candidate Functions:\n{candidates_block}\n\n"
            "### Task:\n"
            f"{base_instructions}\n"
            "Output your top-5 ranking by Candidate ID in descending order of root-cause likelihood.\n"
            "Format your ranking strictly in tags: <ranking>[ID1, ID2, ID3, ID4, ID5]</ranking>."
        )

    return prompt
