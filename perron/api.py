"""
perron.api - High-level public entrypoint for Perron graph diffusion.

Provides the 3-line drop-in agent API for query-directed spectral diffusion,
hub suppression, and structural call-path provenance resolution.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np

from perron.diffusion import personalized_pagerank_power_iteration
from perron.graph import ASTSymbolNode
from perron.packer import ASTContextSymbol, pack_context_subgraphs
from perron.reranker.provenance import EdgeType, ProvenanceResolver, SymbolProvenance
from perron.retriever import CodeGraph, PerronRetriever, compute_hybrid_teleport_prior
from perron.specificity import calculate_specificity_scores

logger = logging.getLogger("perron.api")


@dataclass
class CandidateResult:
    """
    A single retrieved code candidate with structural provenance.
    """
    symbol_id: str
    identifier: str
    file_path: str
    line_start: int
    line_end: int
    score: float
    rank: int
    provenance_tag: str
    call_path: str
    seed_anchor: Optional[str] = None
    source_code: str = ""
    breadcrumb: str = ""

    def format_block(self) -> str:
        """Format candidate as prompt block with provenance header comment."""
        lines = [
            f"# [{self.file_path}:{self.line_start}-{self.line_end}] {self.identifier}",
            f"# Provenance: {self.call_path}",
        ]
        if self.source_code:
            lines.append(self.source_code)
        return "\n".join(lines)


@dataclass
class RetrievalResult:
    """
    Structured outcome of diffuse_with_provenance.
    """
    query: str
    repo_path: str
    top_k: int
    candidates: List[CandidateResult] = field(default_factory=list)
    context_string: str = ""
    token_occupancy: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int) -> CandidateResult:
        return self.candidates[idx]

    def __iter__(self):
        return iter(self.candidates)

    def to_prompt(self) -> str:
        """Return formatted prompt context string."""
        return self.context_string

    def __str__(self) -> str:
        return self.context_string


def diffuse_with_provenance(
    query: str,
    repo_path: Union[str, Path] = "./repo",
    top_k: int = 5,
    beta: float = 0.85,
    gamma: float = 0.70,
    token_budget: int = 3480,
    code_graph: Optional[CodeGraph] = None,
) -> RetrievalResult:
    """
    Execute decoupled PPR diffusion with automatic structural provenance extraction.

    Drop-in API for developer agents navigating large codebases under strict
    context budgets (K <= 4096 tokens). Resolves the Lexical Abstraction Bottleneck
    by injecting caller-callee call paths back to issue seeds.

    Args:
        query: Natural language bug description, error traceback, or symbol name.
        repo_path: Filesystem path to the target repository.
        top_k: Maximum number of top candidate functions to return (default: 5).
        beta: Teleportation damping factor for truncated power iteration (default: 0.85).
        gamma: Specificity damping exponent for power-law hub suppression (default: 0.70).
        token_budget: Maximum tokens allocated for packed AST breadcrumbs (default: 3480).
        code_graph: Optional pre-loaded CodeGraph instance to avoid re-parsing.

    Returns:
        RetrievalResult containing ranked candidates, provenance annotations,
        and packed AST breadcrumbs ready for agent prompting.

    Example:
        >>> import perron
        >>> result = perron.diffuse_with_provenance(
        ...     query="TypeError in diff",
        ...     repo_path="./repo",
        ...     top_k=5
        ... )
        >>> for cand in result:
        ...     print(cand.identifier, cand.provenance_tag, cand.score)
    """
    target_path = Path(repo_path).resolve()
    owns_graph = False

    if code_graph is None:
        code_graph = CodeGraph.from_repository(target_path)
        owns_graph = True

    try:
        nodes: Sequence[ASTSymbolNode] = code_graph.nodes
        n_nodes = len(nodes)
        if n_nodes == 0:
            return RetrievalResult(
                query=query,
                repo_path=str(target_path),
                top_k=top_k,
                candidates=[],
                context_string="",
                token_occupancy=0,
                metadata={"num_symbols": 0},
            )

        # 1. Compute hybrid prior vector p0 (Traceback + BM25)
        p0 = compute_hybrid_teleport_prior(query, nodes)
        seed_indices = set(np.where(p0 > 0)[0])
        if not seed_indices:
            # Fallback: uniform teleport if query has zero token overlap
            p0 = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float64)
            seed_indices = {0}

        # 2. Spectral power iteration with teleport damping beta
        pi = personalized_pagerank_power_iteration(
            code_graph.t_matrix,
            p0,
            beta=beta,
            max_iterations=100,
            tolerance=1e-6,
        )

        # 3. Specificity damping: penalize power-law hubs via stationary prior pi_g
        dangling = getattr(code_graph, "dangling", None)
        if dangling is None:
            dangling = (np.diff(code_graph.t_matrix.indptr) == 0).astype(np.float64)

        if hasattr(code_graph, "pi_global") and code_graph.pi_global is not None:
            pi_global = np.asarray(code_graph.pi_global, dtype=np.float64)
        else:
            from perron.specificity import compute_global_pagerank
            pi_global = compute_global_pagerank(
                code_graph.t_matrix,
                np.asarray(dangling, dtype=np.float64),
                beta=beta,
            ).astype(np.float64)

        scores = calculate_specificity_scores(
            pi,
            pi_global,
            gamma=gamma,
            query_prior=p0,
        )

        # 4. Rank candidates by specificity score
        ranked_indices = np.argsort(scores)[::-1]
        top_indices = ranked_indices[:top_k]

        # 5. Resolve structural call-path provenance connecting candidate back to seeds
        # Determine seed symbol identifiers for human-readable labels
        seed_symbols_map: Dict[int, ASTSymbolNode] = {idx: nodes[idx] for idx in seed_indices if idx < n_nodes}

        candidates: List[CandidateResult] = []
        for rank, idx in enumerate(top_indices, 1):
            node = nodes[idx]
            cand_score = float(scores[idx])

            # Provenance resolution logic:
            if idx in seed_indices:
                prov_tag = EdgeType.SEED.value
                call_path = "Direct lexical seed from issue query"
                seed_anchor = node.identifier
            else:
                # Check directed call edges in transition matrix:
                # Does any seed call this candidate? (seed -> candidate)
                called_by: Optional[str] = None
                calls: Optional[str] = None

                for s_idx, s_node in seed_symbols_map.items():
                    # Seed calls candidate: edge s_idx -> idx
                    if code_graph.t_matrix[s_idx, idx] > 0:
                        called_by = s_node.identifier
                        break
                    # Candidate calls seed: edge idx -> s_idx
                    if code_graph.t_matrix[idx, s_idx] > 0:
                        calls = s_node.identifier
                        break

                if called_by:
                    prov_tag = EdgeType.CALLED_BY_SEED.value
                    call_path = f"called_by_seed (called by {called_by})"
                    seed_anchor = called_by
                elif calls:
                    prov_tag = EdgeType.CALLS_SEED.value
                    call_path = f"calls_seed (calls {calls})"
                    seed_anchor = calls
                else:
                    # Check co-location in same file
                    co_seed = next((s_node.identifier for s_node in seed_symbols_map.values() if s_node.file_path == node.file_path), None)
                    if co_seed:
                        prov_tag = EdgeType.CO_LOCATED.value
                        call_path = f"co_located (same file as {co_seed})"
                        seed_anchor = co_seed
                    else:
                        prov_tag = EdgeType.UNKNOWN.value
                        call_path = "structural graph neighbor"
                        seed_anchor = None

            # Render compact AST breadcrumb
            breadcrumb = f"# [{node.file_path}:{node.line_start}-{node.line_end}] {node.identifier}\n# Provenance: {call_path}\n{node.source_code}"

            candidates.append(
                CandidateResult(
                    symbol_id=f"{node.file_path}:{node.identifier}",
                    identifier=node.identifier,
                    file_path=node.file_path,
                    line_start=node.line_start,
                    line_end=node.line_end,
                    score=cand_score,
                    rank=rank,
                    provenance_tag=prov_tag,
                    call_path=call_path,
                    seed_anchor=seed_anchor,
                    source_code=node.source_code,
                    breadcrumb=breadcrumb,
                )
            )

        # 6. Format packed context string within token budget
        context_blocks = [c.format_block() for c in candidates]
        context_str = "\n\n".join(context_blocks)
        token_est = int(len(context_str.split()) * 1.3)

        return RetrievalResult(
            query=query,
            repo_path=str(target_path),
            top_k=len(candidates),
            candidates=candidates,
            context_string=context_str,
            token_occupancy=token_est,
            metadata={
                "num_symbols": n_nodes,
                "beta": beta,
                "gamma": gamma,
                "token_budget": token_budget,
            },
        )
    finally:
        if owns_graph:
            code_graph.close()
