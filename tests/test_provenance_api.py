"""
Unit tests for perron.diffuse_with_provenance drop-in API.
"""

from pathlib import Path
import tempfile
import pytest

import perron
from perron.api import CandidateResult, RetrievalResult


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a minimal Python repository with call and file relationships."""
    repo = tmp_path / "sample_repo"
    repo.mkdir()

    # Module A: caller
    mod_a = repo / "module_a.py"
    mod_a.write_text(
        "from module_b import helper_fn\n\n"
        "def main_entrypoint():\n"
        "    '''Main service entrypoint.'''\n"
        "    return helper_fn()\n",
        encoding="utf-8",
    )

    # Module B: callee
    mod_b = repo / "module_b.py"
    mod_b.write_text(
        "def helper_fn():\n"
        "    '''Helper function called by main_entrypoint.'''\n"
        "    return 42\n\n"
        "def unrelated_fn():\n"
        "    '''Independent function.'''\n"
        "    return 0\n",
        encoding="utf-8",
    )

    return repo


def test_diffuse_with_provenance_basic(sample_repo: Path):
    result = perron.diffuse_with_provenance(
        query="main_entrypoint service error",
        repo_path=sample_repo,
        top_k=3,
        beta=0.85,
        gamma=0.70,
    )

    assert isinstance(result, RetrievalResult)
    assert len(result) > 0
    assert result.top_k == len(result.candidates)
    assert result.metadata["beta"] == 0.85
    assert result.metadata["gamma"] == 0.70

    # Top candidate should be main_entrypoint or helper_fn
    top_cand = result[0]
    assert isinstance(top_cand, CandidateResult)
    assert top_cand.identifier in ("main_entrypoint", "helper_fn")
    assert top_cand.score > 0.0
    assert top_cand.rank == 1
    assert len(top_cand.breadcrumb) > 0
    assert "# Provenance:" in top_cand.breadcrumb


def test_diffuse_with_provenance_call_relation(sample_repo: Path):
    result = perron.diffuse_with_provenance(
        query="main_entrypoint",
        repo_path=sample_repo,
        top_k=5,
    )

    identifiers = [c.identifier for c in result]
    assert "main_entrypoint" in identifiers

    # Find helper_fn if retrieved
    helper = next((c for c in result if c.identifier == "helper_fn"), None)
    if helper:
        assert helper.provenance_tag in ("called_by_seed", "calls_seed", "co_located", "structural graph neighbor", "seed")
        assert len(helper.call_path) > 0


def test_diffuse_with_provenance_prompt_formatting(sample_repo: Path):
    result = perron.diffuse_with_provenance(
        query="helper_fn",
        repo_path=sample_repo,
        top_k=2,
    )

    prompt = result.to_prompt()
    assert isinstance(prompt, str)
    assert str(result) == prompt
    assert "# [" in prompt
    assert "# Provenance:" in prompt


def test_diffuse_with_provenance_empty_repo(tmp_path: Path):
    empty_dir = tmp_path / "empty_repo"
    empty_dir.mkdir()

    result = perron.diffuse_with_provenance(
        query="test",
        repo_path=empty_dir,
        top_k=5,
    )

    assert len(result) == 0
    assert result.context_string == ""
