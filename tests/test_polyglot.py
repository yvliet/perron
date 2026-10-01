"""
Unit and Integration Tests for Perron Polyglot Tree-sitter Extraction.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np

try:
    from perron.polyglot.treesitter import TreeSitterExtractor, TREE_SITTER_AVAILABLE
    from perron.matrix import build_static_transition_matrix
    from perron.diffusion import personalized_pagerank_power_iteration
    from perron.specificity import calculate_specificity_scores
except ImportError:
    TREE_SITTER_AVAILABLE = False


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter optional dependency not installed")
def test_typescript_extraction_and_diffusion():
    ts_source = """
    export interface ILogger {
        log(msg: string): void;
    }

    export class ConsoleLogger implements ILogger {
        log(msg: string): void {
            console.log(msg);
        }
    }

    export class OrderService {
        private logger: ILogger;

        constructor(logger: ILogger) {
            this.logger = logger;
        }

        processOrder(orderId: number): boolean {
            this.logger.log("Processing order");
            return this.validate(orderId);
        }

        validate(orderId: number): boolean {
            return orderId > 0;
        }
    }
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        ts_file = repo_path / "service.ts"
        ts_file.write_text(ts_source, encoding="utf-8")

        extractor = TreeSitterExtractor()
        symbols = extractor.extract_file(ts_file, repo_path)

        identifiers = {s.identifier for s in symbols}
        assert "ILogger" in identifiers
        assert "ConsoleLogger" in identifiers
        assert "OrderService" in identifiers
        assert "OrderService.processOrder" in identifiers
        assert "OrderService.validate" in identifiers

        process_order_node = next(s for s in symbols if s.identifier == "OrderService.processOrder")
        assert "log" in process_order_node.calls
        assert "validate" in process_order_node.calls

        t_matrix, symbol_index_map = build_static_transition_matrix(symbols)
        assert t_matrix.shape[0] == len(symbols)
        assert t_matrix.nnz > 0

        p0 = np.zeros(len(symbols), dtype=np.float64)
        p0[symbol_index_map["OrderService.processOrder"]] = 1.0

        pi = personalized_pagerank_power_iteration(t_matrix, p0, beta=0.85)
        assert np.isclose(np.sum(pi), 1.0)

        degrees = np.diff(t_matrix.indptr).astype(np.float64)
        scores = calculate_specificity_scores(pi, degrees, gamma=0.70)
        assert scores[symbol_index_map["OrderService.processOrder"]] > 0.0
