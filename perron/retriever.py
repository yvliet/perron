"""
Perron Hybrid Information Retrieval and Teleportation Prior Engine.

Integrates deterministic Python traceback anchor extraction with a zero-dependency
Okapi BM25 full-text index across symbol names, docstrings, and source code bodies.
Implements candidate min-max normalization with temperature calibration to eliminate
Dirac delta probability collapse during personalized PageRank diffusion.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from perron.diffusion import (
    compute_softmax_teleport_prior,
    personalized_pagerank_power_iteration,
)
from perron.graph import (
    ASTSymbolNode,
    compile_and_save_repository_graph,
    extract_repository_graph,
)
from perron.matrix import (
    close_mmap_csr,
    load_mmap_csr,
)
from perron.packer import ASTContextSymbol, pack_context_subgraphs
from perron.specificity import calculate_specificity_scores, compute_global_pagerank

# Standard Python and English stop words to filter out common tokens
# preventing false-positive noise amplification in lexical indexing.
CODE_STOP_WORDS: Set[str] = {
    "def", "class", "self", "cls", "return", "for", "in", "import", "from",
    "as", "pass", "with", "try", "except", "finally", "raise", "while",
    "yield", "lambda", "if", "else", "elif", "not", "and", "or", "is",
    "true", "false", "none", "to", "the", "of", "a", "an", "this", "that",
    "it", "on", "at", "by", "be", "are", "was", "were", "do", "does", "did",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class TracebackAnchor:
    """
    Extracted stack trace frame anchor.
    """
    file_path: str
    line_number: int
    function_name: str


class TracebackParser:
    """
    Deterministic Python traceback parser extracting file, line, and function coordinates.
    """

    TRACEBACK_PATTERN = re.compile(
        r'File\s+["\'](?P<file>[^"\']+)["\'],\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>[^\n]+))?'
    )

    @classmethod
    def extract_anchors(cls, text: str) -> List[TracebackAnchor]:
        """
        Parse all traceback frames present in issue descriptions or logs.
        """
        anchors: List[TracebackAnchor] = []
        for match in cls.TRACEBACK_PATTERN.finditer(text):
            file_str = match.group("file").strip()
            line_str = match.group("line").strip()
            func_str = (match.group("func") or "").strip()

            # Sanitize function name if module or listcomp
            if not func_str or func_str.startswith("<"):
                func_str = ""

            try:
                line_no = int(line_str)
            except ValueError:
                continue

            anchors.append(
                TracebackAnchor(
                    file_path=file_str.replace("\\", "/"),
                    line_number=line_no,
                    function_name=func_str,
                )
            )
        return anchors


class OkapiBM25Index:
    """
    Zero-dependency Okapi BM25 index over ASTSymbolNode documents.
    Indexes qualified names, file paths, docstrings, and code bodies.
    """

    def __init__(
        self,
        symbols: List[ASTSymbolNode],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.num_docs = len(symbols)
        self.doc_lengths: np.ndarray = np.zeros(self.num_docs, dtype=np.float64)
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}

        total_length = 0
        for i, sym in enumerate(symbols):
            tokens = self._tokenize_symbol(sym)
            length = len(tokens)
            self.doc_lengths[i] = length
            total_length += length

            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)

            for token in tf:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = float(total_length) / max(1, self.num_docs)

        # Precompute Robertson-Sparck Jones IDF with smoothing
        self.idf: Dict[str, float] = {}
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log(
                (self.num_docs - df + 0.5) / (df + 0.5) + 1.0
            )

    @staticmethod
    def _tokenize_symbol(sym: ASTSymbolNode) -> List[str]:
        """
        Tokenize a symbol node, weighting name and path tokens higher than code body.
        """
        tokens: List[str] = []

        # Name tokens (boosted 3x)
        name = getattr(sym, "qualified_name", getattr(sym, "name", ""))
        name_tokens = [
            t.lower() for t in TOKEN_PATTERN.findall(name)
            if t.lower() not in CODE_STOP_WORDS and len(t) >= 2
        ]
        tokens.extend(name_tokens * 3)

        # File path tokens (boosted 2x)
        path = getattr(sym, "file_path", "")
        path_tokens = [
            t.lower() for t in TOKEN_PATTERN.findall(path)
            if t.lower() not in CODE_STOP_WORDS and len(t) >= 2
        ]
        tokens.extend(path_tokens * 2)

        # Docstring tokens (if available)
        doc = getattr(sym, "docstring", getattr(sym, "class_docstring", None))
        if doc:
            doc_tokens = [
                t.lower() for t in TOKEN_PATTERN.findall(doc)
                if t.lower() not in CODE_STOP_WORDS and len(t) >= 2
            ]
            tokens.extend(doc_tokens)

        # Code body tokens
        if sym.code:
            code_tokens = [
                t.lower() for t in TOKEN_PATTERN.findall(sym.code)
                if t.lower() not in CODE_STOP_WORDS and len(t) >= 2
            ]
            tokens.extend(code_tokens)

        return tokens

    def query(self, query_text: str) -> np.ndarray:
        """
        Compute BM25 relevance scores for all indexed symbols.
        Returns a 1D numpy array of length num_docs.
        """
        scores = np.zeros(self.num_docs, dtype=np.float64)
        query_tokens = [
            t.lower() for t in TOKEN_PATTERN.findall(query_text)
            if t.lower() not in CODE_STOP_WORDS and len(t) >= 2
        ]

        if not query_tokens or self.num_docs == 0:
            return scores

        for qtok in query_tokens:
            if qtok not in self.idf:
                continue
            token_idf = self.idf[qtok]

            # Iterate over documents containing qtok
            for doc_id, tf_dict in enumerate(self.doc_term_freqs):
                if qtok in tf_dict:
                    freq = tf_dict[qtok]
                    doc_len = self.doc_lengths[doc_id]
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    scores[doc_id] += token_idf * ((freq * (self.k1 + 1.0)) / denom)

        return scores


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize candidate scores to [0.0, 1.0] with an explicit division-by-zero guard.
    When candidate scores are identical, returns an array of zeros.
    """
    if scores.size == 0:
        return scores
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    denom = s_max - s_min
    if denom < 1e-12:
        return np.zeros_like(scores)
    return (scores - s_min) / denom


def compute_traceback_prior(
    anchors: List[TracebackAnchor],
    symbols: List[ASTSymbolNode],
) -> np.ndarray:
    """
    Construct a sparse prior vector placing probability mass directly on symbols
    matching traceback frames (file path and line number coordinates).
    Falls back to file-level distribution if specific symbol is unmapped.
    """
    p_trace = np.zeros(len(symbols), dtype=np.float64)
    if not anchors or not symbols:
        return p_trace

    # Map file paths to lists of symbol indices
    file_to_indices: Dict[str, List[int]] = {}
    for i, sym in enumerate(symbols):
        norm_path = sym.file_path.replace("\\", "/").lower()
        file_to_indices.setdefault(norm_path, []).append(i)

    hit_indices: Set[int] = set()

    for anchor in anchors:
        anchor_file = anchor.file_path.lower()
        matched_indices: List[int] = []

        # Find matching repository file by suffix
        for repo_file, indices in file_to_indices.items():
            if anchor_file.endswith(repo_file) or repo_file.endswith(anchor_file):
                matched_indices.extend(indices)
                break

        if not matched_indices:
            # Try file basename match
            anchor_base = Path(anchor.file_path).name.lower()
            for repo_file, indices in file_to_indices.items():
                if Path(repo_file).name.lower() == anchor_base:
                    matched_indices.extend(indices)
                    break

        if not matched_indices:
            continue

        # Check line number containment
        line_matched: List[int] = []
        for idx in matched_indices:
            sym = symbols[idx]
            # Match within span with a 5-line margin of tolerance
            if (sym.start_line - 5) <= anchor.line_number <= (sym.end_line + 5):
                line_matched.append(idx)

        if line_matched:
            hit_indices.update(line_matched)
        elif anchor.function_name:
            # Check function name match in file
            fn_matched = [
                idx for idx in matched_indices
                if getattr(symbols[idx], "qualified_name", getattr(symbols[idx], "name", "")).lower().endswith(anchor.function_name.lower())
            ]
            if fn_matched:
                hit_indices.update(fn_matched)
            else:
                # File-level fallback: partition mass across all symbols in file
                hit_indices.update(matched_indices)
        else:
            hit_indices.update(matched_indices)

    if hit_indices:
        mass_per_hit = 1.0 / len(hit_indices)
        for idx in hit_indices:
            p_trace[idx] = mass_per_hit

    return p_trace


def compute_hybrid_teleport_prior(
    query_text: str,
    symbols: List[ASTSymbolNode],
    bm25_index: Optional[OkapiBM25Index] = None,
    temperature: float = 0.15,
    top_k: int = 25,
) -> np.ndarray:
    """
    Evaluate the query-directed hybrid teleportation prior p_0.
    Blends traceback frame coordinates with candidate min-max normalized Okapi BM25 scores.
    Strict Invariant: sum(p_0) == 1.0, entries in [0.0, 1.0].
    """
    num_nodes = len(symbols)
    if num_nodes == 0:
        return np.array([], dtype=np.float64)

    # 1. Evaluate BM25 relevance scores
    if bm25_index is None:
        bm25_index = OkapiBM25Index(symbols)
    raw_bm25 = bm25_index.query(query_text)

    # 2. Candidate pool selection (top_k)
    k = min(top_k, num_nodes)
    p_bm25 = np.zeros(num_nodes, dtype=np.float64)

    if k > 0:
        top_indices = np.argsort(raw_bm25)[-k:]
        candidate_scores = raw_bm25[top_indices]

        # 3. Min-Max normalization with epsilon guard
        norm_scores = min_max_normalize(candidate_scores)

        # 4. Shift-invariant softmax over candidate pool
        safe_temp = max(float(temperature), 1e-4)
        scaled_scores = norm_scores / safe_temp
        c = np.max(scaled_scores)
        exp_scores = np.exp(scaled_scores - c)
        sum_exp = np.sum(exp_scores)
        softmax_weights = exp_scores / sum_exp if sum_exp > 0 else np.full_like(exp_scores, 1.0 / len(exp_scores))
        p_bm25[top_indices] = softmax_weights

    # 5. Extract and map traceback anchors
    anchors = TracebackParser.extract_anchors(query_text)
    p_trace = compute_traceback_prior(anchors, symbols)

    # 6. Convex combination
    if np.sum(p_trace) > 0.0:
        p_0 = 0.50 * p_trace + 0.50 * p_bm25
    else:
        p_0 = p_bm25

    # Strictly re-normalize to ensure numerical unity
    total = np.sum(p_0)
    if total > 0.0:
        p_0 /= total
    else:
        p_0 = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)

    return p_0


logger = logging.getLogger("perron.retriever")


class CodeGraph:
    """Encapsulates repository AST symbols and memory-mapped CSR transition matrix."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        nodes: Optional[List[Any]] = None,
        t_matrix: Any = None,
        symbols: Optional[Sequence[Any]] = None,
        metadata: Optional[dict] = None,
        dangling: Optional[np.ndarray] = None,
        pi_global: Optional[np.ndarray] = None,
        symbol_index: Optional[dict] = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        if nodes is not None:
            self.nodes = list(nodes)
        elif symbols is not None:
            self.nodes = list(symbols)
        else:
            self.nodes = []

        self.t_matrix = t_matrix
        self.symbols = list(symbols) if symbols is not None else self.nodes
        self.metadata = metadata if metadata is not None else {}
        self.dangling = dangling
        self.pi_global = pi_global
        self.symbol_index = (
            symbol_index
            if symbol_index is not None
            else {
                getattr(s, "qualified_name", getattr(s, "name", str(s))): i
                for i, s in enumerate(self.symbols)
            }
        )
        self.n_symbols = len(self.symbols)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_directory(
        cls,
        repo_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        force_recompile: bool = False,
    ) -> CodeGraph:
        """Compile or load a repository call graph from disk."""
        repo_path = Path(repo_dir).resolve()
        cache_path = (
            Path(cache_dir).resolve()
            if cache_dir
            else repo_path / ".perron_cache"
        )

        has_cache = (cache_path / "indptr.npy").exists() or (cache_path / "t_matrix_indptr.npy").exists()
        if not has_cache or force_recompile:
            logger.info(f"Compiling repository AST graph for {repo_path}...")
            t_matrix, dangling, canonical_to_id, id_to_canonical, nodes = compile_and_save_repository_graph(repo_path, cache_path)
            # Create alias file t_matrix_indptr.npy if indptr.npy exists
            indptr_path = cache_path / "indptr.npy"
            t_indptr_path = cache_path / "t_matrix_indptr.npy"
            if indptr_path.exists() and not t_indptr_path.exists():
                try:
                    import shutil
                    shutil.copy2(indptr_path, t_indptr_path)
                except Exception:
                    pass
            pi_global_path = cache_path / "pi_global.npy"
            if pi_global_path.exists():
                try:
                    pi_global = np.load(pi_global_path, mmap_mode="r")
                except Exception:
                    pi_global = compute_global_pagerank(t_matrix, dangling)
            else:
                pi_global = compute_global_pagerank(t_matrix, dangling)
            symbols = nodes
            metadata = {
                "symbols_count": len(symbols),
                "shape": list(t_matrix.shape),
                "nnz": int(t_matrix.nnz),
            }
        else:
            logger.info(f"Loading existing cache from {cache_path}...")
            sym_file = cache_path / "id_to_symbol.json"
            if not sym_file.exists():
                sym_file = cache_path / "symbols.json"
            if sym_file.exists():
                with open(sym_file, "r", encoding="utf-8") as f:
                    raw_syms = json.load(f)
                    nodes = [ASTSymbolNode(**item) for item in raw_syms]
            else:
                nodes_res = extract_repository_graph(repo_path)
                nodes = nodes_res[0] if isinstance(nodes_res, tuple) else nodes_res
            t_matrix, dangling, node_to_id, id_to_node = load_mmap_csr(cache_path)
            pi_global_path = cache_path / "pi_global.npy"
            if pi_global_path.exists():
                try:
                    pi_global = np.load(pi_global_path, mmap_mode="r")
                except Exception:
                    pi_global = compute_global_pagerank(t_matrix, dangling)
            else:
                pi_global = compute_global_pagerank(t_matrix, dangling)
                try:
                    np.save(pi_global_path, pi_global.astype(np.float64))
                except Exception:
                    pass
            symbols = nodes
            metadata = {
                "symbols_count": len(symbols),
                "shape": list(t_matrix.shape),
                "nnz": int(t_matrix.nnz),
            }

        return cls(cache_path, nodes, t_matrix, symbols, metadata, dangling, pi_global)

    def close(self) -> None:
        """Close memory-mapped array buffers safely."""
        close_mmap_csr(self.t_matrix)

    def __enter__(self) -> CodeGraph:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class PerronRetriever:
    """High-throughput query-directed spectral diffusion retriever."""

    def __init__(
        self,
        code_graph: CodeGraph,
        gamma: float = 0.70,
        beta: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        self.graph = code_graph
        self.gamma = gamma
        self.beta = beta
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Precompute or load stationary global PageRank vector pi_g for smoothed PMI hub damping
        # Dynamically recompute if self.beta departs from baseline beta=0.85 to maintain spectral alignment
        if (
            hasattr(self.graph, "pi_global")
            and self.graph.pi_global is not None
            and abs(self.beta - 0.85) < 1e-4
        ):
            self.pi_global = np.asarray(self.graph.pi_global, dtype=np.float64)
        else:
            dangling = getattr(self.graph, "dangling", None)
            if dangling is None:
                dangling = (np.diff(self.graph.t_matrix.indptr) == 0).astype(np.float64)
            self.pi_global = compute_global_pagerank(
                self.graph.t_matrix,
                np.asarray(dangling, dtype=np.float64),
                beta=self.beta,
                max_iter=self.max_iterations,
                tol=self.tolerance,
            ).astype(np.float64)

        # Static degrees vector for structural bounds and backward compatibility
        self.degrees = np.diff(self.graph.t_matrix.indptr).astype(np.float64)

    def query(
        self,
        query_text: str,
        max_tokens: int = 4096,
        return_symbols: bool = False,
    ) -> Union[str, List[ASTContextSymbol]]:
        """
        Execute query-directed spectral diffusion and return packed AST code slices.
        """
        # 1. Compute hybrid prior vector p0
        p0 = compute_hybrid_teleport_prior(query_text, self.graph.nodes)

        # 2. Spectral power iteration with teleport damping beta
        pi = personalized_pagerank_power_iteration(
            self.graph.t_matrix,
            p0,
            beta=self.beta,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

        # 3. Specificity damping: penalize power-law hubs via stationary distribution pi_g
        scores = calculate_specificity_scores(
            pi,
            self.pi_global,
            gamma=self.gamma,
            query_prior=p0,
        )

        # 4. AST subgraph context packing within token budget K
        packed_symbols, context_str = pack_context_subgraphs(
            self.graph.nodes,
            scores,
            adjacency_matrix=self.graph.t_matrix,
            token_budget=max_tokens,
        )

        if return_symbols:
            return packed_symbols

        formatted_chunks = []
        for sym in packed_symbols:
            header = f"# [{sym.file_path}:{sym.line_start}-{sym.line_end}] {sym.identifier}"
            formatted_chunks.append(f"{header}\n{sym.source_code}")

        return "\n\n".join(formatted_chunks) if formatted_chunks else context_str

    def retrieve(
        self,
        query_text: str,
        max_tokens: int = 4096,
    ) -> List[ASTContextSymbol]:
        """Convenience alias returning packed AST symbols for the query within token budget."""
        return self.query(query_text=query_text, max_tokens=max_tokens, return_symbols=True)

