"""
Perron: Context-Budgeted AST Subgraph Slicing and Graph Retrieval for Developer Agents.
"""

from __future__ import annotations

from perron.matrix import (
    build_static_transition_matrix,
    load_mmap_csr,
    save_mmap_csr,
    close_mmap_csr,
)
from perron.diffusion import (
    compute_softmax_teleport_prior,
    personalized_pagerank_power_iteration,
)
from perron.specificity import (
    compute_global_pagerank,
    calculate_specificity_scores,
)
from perron.packer import (
    pack_context_subgraphs,
    ASTContextSymbol,
)
from perron.editor import (
    apply_symbol_edit,
    apply_multi_file_patch,
    SymbolEditSpec,
)
from perron.patch import (
    compute_git_patch,
    generate_unified_diff,
)
from perron.tester import (
    run_targeted_test,
)
from perron.graph import (
    ASTSymbolNode,
    extract_repository_graph,
    compile_and_save_repository_graph,
)
from perron.agent import (
    PerronAgent,
    TrajectoryResult,
    TurnTelemetry,
)
from perron.retriever import (
    CodeGraph,
    PerronRetriever,
)

__all__ = [
    "build_static_transition_matrix",
    "load_mmap_csr",
    "save_mmap_csr",
    "close_mmap_csr",
    "compute_softmax_teleport_prior",
    "personalized_pagerank_power_iteration",
    "compute_global_pagerank",
    "calculate_specificity_scores",
    "pack_context_subgraphs",
    "ASTContextSymbol",
    "apply_symbol_edit",
    "apply_multi_file_patch",
    "SymbolEditSpec",
    "compute_git_patch",
    "generate_unified_diff",
    "run_targeted_test",
    "ASTSymbolNode",
    "extract_repository_graph",
    "compile_and_save_repository_graph",
    "PerronAgent",
    "TrajectoryResult",
    "TurnTelemetry",
    "CodeGraph",
    "PerronRetriever",
]

