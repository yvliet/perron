"""
Unit tests for Perron Hybrid Information Retrieval and Prior Construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from perron.graph import ASTSymbolNode
from perron.retriever import (
    TracebackAnchor,
    TracebackParser,
    OkapiBM25Index,
    min_max_normalize,
    compute_traceback_prior,
    compute_hybrid_teleport_prior,
)


def _make_dummy_symbols() -> list[ASTSymbolNode]:
    return [
        ASTSymbolNode(
            node_id=0,
            file_path="requests/models.py",
            qualified_name="PreparedRequest.prepare_url",
            symbol_type="method",
            start_line=350,
            end_line=410,
            code="def prepare_url(self, url, params):\n    self.url = url\n",
            token_count=15,
            parent_class="PreparedRequest",
            docstring="Prepares the given HTTP URL.",
        ),
        ASTSymbolNode(
            node_id=1,
            file_path="requests/models.py",
            qualified_name="PreparedRequest.prepare_headers",
            symbol_type="method",
            start_line=415,
            end_line=450,
            code="def prepare_headers(self, headers):\n    self.headers = headers\n",
            token_count=12,
            parent_class="PreparedRequest",
            docstring="Prepares the given HTTP headers.",
        ),
        ASTSymbolNode(
            node_id=2,
            file_path="requests/sessions.py",
            qualified_name="Session.send",
            symbol_type="method",
            start_line=600,
            end_line=650,
            code="def send(self, request, **kwargs):\n    return self.adapter.send(request)\n",
            token_count=18,
            parent_class="Session",
            docstring="Send a given PreparedRequest.",
        ),
        ASTSymbolNode(
            node_id=3,
            file_path="requests/utils.py",
            qualified_name="to_native_string",
            symbol_type="function",
            start_line=20,
            end_line=35,
            code="def to_native_string(string, encoding='ascii'):\n    return str(string)\n",
            token_count=14,
            parent_class=None,
            docstring="Convert string to native representation.",
        ),
    ]


def test_traceback_parser_extracts_frames():
    tb_text = """
    Traceback (most recent call last):
      File "/home/user/requests/sessions.py", line 625, in send
        r = adapter.send(request, **kwargs)
      File "/home/user/requests/models.py", line 375, in prepare_url
        raise InvalidURL("Invalid URL format")
    InvalidURL: Invalid URL format
    """
    anchors = TracebackParser.extract_anchors(tb_text)
    assert len(anchors) == 2
    assert anchors[0].file_path.endswith("requests/sessions.py")
    assert anchors[0].line_number == 625
    assert anchors[0].function_name == "send"

    assert anchors[1].file_path.endswith("requests/models.py")
    assert anchors[1].line_number == 375
    assert anchors[1].function_name == "prepare_url"


def test_traceback_parser_handles_empty_or_malformed():
    assert TracebackParser.extract_anchors("No traceback here, just a bug report.") == []
    tb_malformed = 'File "", line abc, in ?'
    assert TracebackParser.extract_anchors(tb_malformed) == []


def test_min_max_normalize_epsilon_guard():
    # Constant scores (zero denominator)
    constant_scores = np.array([5.0, 5.0, 5.0])
    norm = min_max_normalize(constant_scores)
    assert np.all(norm == 0.0)

    # All zeros
    zero_scores = np.zeros(4)
    assert np.all(min_max_normalize(zero_scores) == 0.0)

    # Standard distribution
    scores = np.array([10.0, 20.0, 30.0])
    norm = min_max_normalize(scores)
    np.testing.assert_allclose(norm, [0.0, 0.5, 1.0])


def test_bm25_index_relevance():
    symbols = _make_dummy_symbols()
    index = OkapiBM25Index(symbols)

    # Query matching url method
    scores_url = index.query("URL parsing error in prepare_url")
    assert scores_url[0] > scores_url[1]  # prepare_url > prepare_headers
    assert scores_url[0] > scores_url[2]

    # Query matching session send
    scores_session = index.query("Session adapter send fails")
    assert scores_session[2] > scores_session[0]

    # Empty query yields zeros
    scores_empty = index.query("")
    assert np.all(scores_empty == 0.0)


def test_compute_traceback_prior_mapping():
    symbols = _make_dummy_symbols()
    anchors = [
        TracebackAnchor(file_path="requests/models.py", line_number=380, function_name="prepare_url")
    ]
    p_trace = compute_traceback_prior(anchors, symbols)
    assert p_trace[0] == 1.0
    assert np.sum(p_trace) == 1.0
    assert p_trace[1] == 0.0
    assert p_trace[2] == 0.0


def test_compute_traceback_prior_external_fallback():
    symbols = _make_dummy_symbols()
    # Anchor in external unindexed library
    anchors = [
        TracebackAnchor(file_path="asyncio/events.py", line_number=100, function_name="_run")
    ]
    p_trace = compute_traceback_prior(anchors, symbols)
    # When anchor points to external module, sum(p_trace) is 0.0
    assert np.sum(p_trace) == 0.0


def test_compute_hybrid_teleport_prior_unity_and_bounds():
    symbols = _make_dummy_symbols()
    index = OkapiBM25Index(symbols)

    # 1. Query with traceback
    query_with_tb = """
    File "requests/models.py", line 375, in prepare_url
    Exception: Invalid URL format
    """
    p_0 = compute_hybrid_teleport_prior(query_with_tb, symbols, bm25_index=index, temperature=0.15)
    assert np.isclose(np.sum(p_0), 1.0, atol=1e-12)
    assert np.all(p_0 >= 0.0)
    assert np.all(p_0 <= 1.0)
    assert p_0[0] > 0.4  # High probability mass on target node

    # 2. Query without traceback
    query_without_tb = "Invalid URL encoding in prepare_url method"
    p_0_no_tb = compute_hybrid_teleport_prior(query_without_tb, symbols, bm25_index=index, temperature=0.15)
    assert np.isclose(np.sum(p_0_no_tb), 1.0, atol=1e-12)
    assert np.all(p_0_no_tb >= 0.0)
    assert p_0_no_tb[0] > p_0_no_tb[1]

    # 3. Completely unrelated query (zero BM25 match)
    p_0_unrelated = compute_hybrid_teleport_prior("quantum entanglement superconductor", symbols, bm25_index=index)
    assert np.isclose(np.sum(p_0_unrelated), 1.0, atol=1e-12)
    # Falls back to uniform without NaN
    assert not np.any(np.isnan(p_0_unrelated))


def test_compute_hybrid_teleport_prior_zero_temperature_safety():
    """
    Verifies that passing temperature=0.0 or negative values is safely guarded
    and returns a valid non-NaN probability distribution summing to 1.0.
    """
    symbols = _make_dummy_symbols()
    index = OkapiBM25Index(symbols)

    for bad_temp in [0.0, -1.0, 1e-8]:
        p_0 = compute_hybrid_teleport_prior("prepare_url", symbols, bm25_index=index, temperature=bad_temp)
        assert not np.any(np.isnan(p_0)), f"NaN detected for temperature={bad_temp}"
        assert not np.any(np.isinf(p_0)), f"Inf detected for temperature={bad_temp}"
        assert np.isclose(np.sum(p_0), 1.0, atol=1e-12)
        assert np.all(p_0 >= 0.0)


def test_compute_hybrid_teleport_prior_top_k_zero():
    """
    Verifies that top_k=0 does not trigger Python negative slice semantics ([-0:])
    and safely returns valid normalized priors.
    """
    symbols = _make_dummy_symbols()
    index = OkapiBM25Index(symbols)

    # Without traceback, top_k=0 yields uniform fallback without errors
    p_0_uniform = compute_hybrid_teleport_prior("prepare_url", symbols, bm25_index=index, top_k=0)
    assert not np.any(np.isnan(p_0_uniform))
    assert np.isclose(np.sum(p_0_uniform), 1.0, atol=1e-12)
    assert np.allclose(p_0_uniform, 1.0 / len(symbols))

    # With traceback, top_k=0 attributes 100% of mass to traceback anchors
    tb_query = 'File "requests/models.py", line 375, in prepare_url\nException: Failed'
    p_0_tb = compute_hybrid_teleport_prior(tb_query, symbols, bm25_index=index, top_k=0)
    assert not np.any(np.isnan(p_0_tb))
    assert np.isclose(np.sum(p_0_tb), 1.0, atol=1e-12)
    assert p_0_tb[0] == 1.0
    assert p_0_tb[1] == 0.0


