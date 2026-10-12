"""
tests/test_causal_modules.py - Verification for Causal Obfuscation and Graph Encodings.
"""

import ast
import unittest
from perron.reranker.obfuscation import obfuscate_code_snippet, IdentifierObfuscator
from perron.reranker.graph_encodings import (
    GraphEncodingStyle,
    format_candidate_block,
    build_experiment_prompt,
)


class TestCausalModules(unittest.TestCase):
    def test_identifier_obfuscation_syntax_validity(self):
        source = """
def compute_loss(predictions, targets, mask=None):
    error = predictions - targets
    if mask is not None:
        error = error * mask
    return error.sum()
"""
        obf, mapping = obfuscate_code_snippet(source)
        self.assertNotIn("compute_loss", obf)
        self.assertNotIn("predictions", obf)
        self.assertNotIn("targets", obf)
        self.assertIn("fn_1", obf)
        self.assertIn("is not None", obf)
        
        # Verify that obfuscated code is valid Python syntax
        parsed = ast.parse(obf)
        self.assertIsNotNone(parsed)

    def test_identifier_obfuscation_fallback(self):
        # Snippet with truncated syntax
        truncated = "def broken_func(a, b:"
        obf, mapping = obfuscate_code_snippet(truncated)
        self.assertIsInstance(obf, str)
        self.assertNotIn("broken_func", obf)

    def test_graph_encoding_styles(self):
        cands = [
            {"cand_index": 0, "identifier": "seed_func", "file_path": "a.py", "line_start": 1, "line_end": 10, "breadcrumb": "def seed_func(): pass", "is_gold": True},
            {"cand_index": 1, "identifier": "helper_func", "file_path": "b.py", "line_start": 11, "line_end": 20, "breadcrumb": "def helper_func(): pass", "is_gold": False},
        ]

        # E0 Baseline
        b0 = format_candidate_block(cands, style=GraphEncodingStyle.E0_BASELINE)
        self.assertIn("[Candidate ID: 0]", b0)
        self.assertNotIn("Provenance:", b0)
        self.assertNotIn("Call-Graph Edges", b0)

        # E1 Comment
        b1 = format_candidate_block(cands, style=GraphEncodingStyle.E1_COMMENT)
        self.assertIn("Provenance:", b1)

        # E2 Linearized Path
        b2 = format_candidate_block(cands, style=GraphEncodingStyle.E2_LINEARIZED_PATH, seed_identifiers=["seed_func"])
        self.assertIn("[Call Trace]:", b2)

        # E3 Edge List
        b3 = format_candidate_block(cands, style=GraphEncodingStyle.E3_EDGE_LIST, seed_identifiers=["seed_func"])
        self.assertIn("### Repository Call-Graph Edges:", b3)
        self.assertIn("(seed_func, DEFECT_CALLS, seed_func)", b3)

        # E4 Adjacency Map
        b4 = format_candidate_block(cands, style=GraphEncodingStyle.E4_ADJACENCY_MAP, seed_identifiers=["seed_func"])
        self.assertIn("### Symbol Dependency Adjacency:", b4)

    def test_prompt_generation(self):
        prompt_raw = build_experiment_prompt("Broken query", "Cands block", use_thinking=False)
        self.assertNotIn("<thought>", prompt_raw)
        self.assertIn("<ranking>", prompt_raw)

        prompt_thought = build_experiment_prompt("Broken query", "Cands block", use_thinking=True)
        self.assertIn("<thought>", prompt_thought)
        self.assertIn("<ranking>", prompt_thought)


if __name__ == "__main__":
    unittest.main()
