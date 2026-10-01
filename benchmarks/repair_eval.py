"""
Perron Autonomous Agent Repair Evaluation Harness.

Evaluates PerronAgent across archetypal repository bug instances:
1. Context-budgeted subgraph retrieval via query-directed specificity diffusion.
2. Gemma 4 prompt formulation with native thinking mode (<|think|>).
3. Atomic AST-grounded search-and-replace patching.
4. Isolated targeted pytest execution.
5. Telemetry collection and export to benchmarks/run_telemetry.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tempfile
import time
from typing import List, Optional
from perron import PerronAgent, TrajectoryResult
from perron.backends.base import ModelBackend
from perron.backends.replay import OfflineReplayBackend
from benchmarks.telemetry import aggregate_trajectories, save_telemetry


BENCHMARK_CASES = [
    # --------------------------------------------------------------------------
    # Case 1: Django QuerySet filter clone (Resolved in Turn 1)
    # --------------------------------------------------------------------------
    {
        "id": "django_style_query_filter",
        "issue": "QuerySet.filter() drops clone arguments when called with kwargs.",
        "file": "query.py",
        "code": (
            "class QuerySet:\n"
            "    def filter(self, **kwargs):\n"
            "        # Bug: drops filter_args\n"
            "        if not kwargs:\n"
            "            return self\n"
            "        return self._clone(filter_args=kwargs)\n"
            "    def _clone(self, filter_args=None):\n"
            "        return 'cloned_active' if filter_args and filter_args.get('active') else 'cloned_empty'\n"
        ),
        "test_file": "test_query.py",
        "test_code": (
            "from query import QuerySet\n"
            "def test_filter():\n"
            "    qs = QuerySet()\n"
            "    res = qs.filter(active=True)\n"
            "    assert res == 'cloned_active'\n"
        ),
        "model_responses": [
            (
                "<|think|>\n"
                "The issue report states that QuerySet.filter drops clone arguments.\n"
                "In query.py, the filter method checks 'if not kwargs' and returns self._clone(filter_args=kwargs).\n"
                "Looking at the test, filter(active=True) expects 'cloned_active'.\n"
                "</|think|>\n"
                "<edit file=\"query.py\" start_line=\"4\" end_line=\"6\">\n"
                "<old>\n"
                "        if not kwargs:\n"
                "            return self\n"
                "        return self._clone(filter_args=kwargs)\n"
                "</old>\n"
                "<new>\n"
                "        if not kwargs:\n"
                "            return self._clone()\n"
                "        return self._clone(filter_args=kwargs)\n"
                "</new>\n"
                "</edit>\n"
            )
        ],
    },
    # --------------------------------------------------------------------------
    # Case 2: SymPy Polynomial division by zero (Resolved in Turn 1)
    # --------------------------------------------------------------------------
    {
        "id": "sympy_style_polynomial_division",
        "issue": "Poly.div() silently returns 0 on zero divisor instead of raising ZeroDivisionError.",
        "file": "poly.py",
        "code": (
            "class Poly:\n"
            "    def __init__(self, val=10):\n"
            "        self.val = val\n"
            "    def div(self, other):\n"
            "        if other == 0:\n"
            "            return 0\n"
            "        return self.val // other\n"
        ),
        "test_file": "test_poly.py",
        "test_code": (
            "import pytest\n"
            "from poly import Poly\n"
            "def test_zero_div():\n"
            "    p = Poly(10)\n"
            "    with pytest.raises(ZeroDivisionError):\n"
            "        p.div(0)\n"
        ),
        "model_responses": [
            (
                "<|think|>\n"
                "Poly.div(other) returns 0 when other == 0. The specification requires raising ZeroDivisionError.\n"
                "</|think|>\n"
                "<edit file=\"poly.py\" start_line=\"5\" end_line=\"6\">\n"
                "<old>\n"
                "        if other == 0:\n"
                "            return 0\n"
                "</old>\n"
                "<new>\n"
                "        if other == 0:\n"
                "            raise ZeroDivisionError('polynomial division by zero')\n"
                "</new>\n"
                "</edit>\n"
            )
        ],
    },
    # --------------------------------------------------------------------------
    # Case 3: Flask HTTP Header parsing (Resolved in Turn 1)
    # --------------------------------------------------------------------------
    {
        "id": "flask_style_header_parsing",
        "issue": "parse_header() strips parameter attributes and fails to extract charset.",
        "file": "headers.py",
        "code": (
            "def parse_header(raw: str) -> dict:\n"
            "    if not raw:\n"
            "        return {}\n"
            "    parts = raw.split(';')\n"
            "    return {'main': parts[0].strip()}\n"
        ),
        "test_file": "test_headers.py",
        "test_code": (
            "from headers import parse_header\n"
            "def test_parse_header_params():\n"
            "    res = parse_header('text/html; charset=utf-8')\n"
            "    assert res.get('main') == 'text/html'\n"
            "    assert res.get('charset') == 'utf-8'\n"
        ),
        "model_responses": [
            (
                "<|think|>\n"
                "parse_header only parses parts[0] as 'main' and ignores remaining parts.\n"
                "</|think|>\n"
                "<edit file=\"headers.py\" start_line=\"4\" end_line=\"5\">\n"
                "<old>\n"
                "    parts = raw.split(';')\n"
                "    return {'main': parts[0].strip()}\n"
                "</old>\n"
                "<new>\n"
                "    parts = raw.split(';')\n"
                "    result = {'main': parts[0].strip()}\n"
                "    for param in parts[1:]:\n"
                "        if '=' in param:\n"
                "            k, v = param.split('=', 1)\n"
                "            result[k.strip()] = v.strip()\n"
                "    return result\n"
                "</new>\n"
                "</edit>\n"
            )
        ],
    },
    # --------------------------------------------------------------------------
    # Case 4: Requests Retry backoff factor (Resolved in Turn 2 via test feedback)
    # --------------------------------------------------------------------------
    {
        "id": "requests_style_retry_backoff",
        "issue": "Retry.get_backoff() returns 0.0 on attempt=0 when backoff_factor > 0.",
        "file": "retry.py",
        "code": (
            "class Retry:\n"
            "    def __init__(self, backoff_factor: float = 0.5):\n"
            "        self.backoff_factor = backoff_factor\n"
            "    def get_backoff(self, attempt: int) -> float:\n"
            "        if attempt < 0:\n"
            "            return 0.0\n"
            "        return self.backoff_factor * (2 ** (attempt - 1)) if attempt > 0 else 0.0\n"
        ),
        "test_file": "test_retry.py",
        "test_code": (
            "from retry import Retry\n"
            "def test_backoff_zero():\n"
            "    r = Retry(0.5)\n"
            "    assert r.get_backoff(0) == 0.5\n"
            "    assert r.get_backoff(1) == 1.0\n"
        ),
        "model_responses": [
            # Turn 1: imperfect patch, fails assertion r.get_backoff(0) == 0.5
            (
                "<|think|>\n"
                "Attempting to adjust default backoff for attempt 0.\n"
                "</|think|>\n"
                "<edit file=\"retry.py\" start_line=\"6\" end_line=\"7\">\n"
                "<old>\n"
                "        return self.backoff_factor * (2 ** (attempt - 1)) if attempt > 0 else 0.0\n"
                "</old>\n"
                "<new>\n"
                "        return self.backoff_factor * (2 ** (attempt - 1)) if attempt > 0 else 0.1\n"
                "</new>\n"
                "</edit>\n"
            ),
            # Turn 2: receives test failure feedback and fixes formula accurately
            (
                "<|think|>\n"
                "The test failed because get_backoff(0) returned 0.1 instead of 0.5.\n"
                "The correct calculation is self.backoff_factor * (2 ** attempt).\n"
                "</|think>\n"
                "<edit file=\"retry.py\" start_line=\"6\" end_line=\"7\">\n"
                "<old>\n"
                "        return self.backoff_factor * (2 ** (attempt - 1)) if attempt > 0 else 0.1\n"
                "</old>\n"
                "<new>\n"
                "        return self.backoff_factor * (2 ** attempt)\n"
                "</new>\n"
                "</edit>\n"
            ),
        ],
    },
    # --------------------------------------------------------------------------
    # Case 5: Pydantic Field validator bounds (Resolved in Turn 2 via patch recovery)
    # --------------------------------------------------------------------------
    {
        "id": "pydantic_style_field_validator",
        "issue": "validate_age() fails to reject negative age integers.",
        "file": "validator.py",
        "code": (
            "def validate_age(val):\n"
            "    if isinstance(val, int):\n"
            "        return val\n"
            "    raise ValueError('must be integer')\n"
        ),
        "test_file": "test_validator.py",
        "test_code": (
            "import pytest\n"
            "from validator import validate_age\n"
            "def test_age():\n"
            "    assert validate_age(25) == 25\n"
            "    with pytest.raises(ValueError):\n"
            "        validate_age(-5)\n"
        ),
        "model_responses": [
            # Turn 1: patch fails due to target line drift / mismatch
            (
                "<|think|>\n"
                "Updating validate_age to reject negative integers.\n"
                "</|think>\n"
                "<edit file=\"validator.py\" start_line=\"2\" end_line=\"4\">\n"
                "<old>\n"
                "    non_matching_stale_line_block\n"
                "</old>\n"
                "<new>\n"
                "    if isinstance(val, int) and val >= 0:\n"
                "        return val\n"
                "</new>\n"
                "</edit>\n"
            ),
            # Turn 2: receives patch error diagnostic and emits valid matching block
            (
                "<|think|>\n"
                "Patch failed to match target block. Re-grounding to exact AST source.\n"
                "</|think>\n"
                "<edit file=\"validator.py\" start_line=\"2\" end_line=\"4\">\n"
                "<old>\n"
                "    if isinstance(val, int):\n"
                "        return val\n"
                "    raise ValueError('must be integer')\n"
                "</old>\n"
                "<new>\n"
                "    if isinstance(val, int) and val >= 0:\n"
                "        return val\n"
                "    raise ValueError('invalid age: must be non-negative integer')\n"
                "</new>\n"
                "</edit>\n"
            ),
        ],
    },
    # --------------------------------------------------------------------------
    # Case 6: Unresolved - Multi-file Latent Dependency
    # --------------------------------------------------------------------------
    {
        "id": "unresolved_multi_file_latent_dep",
        "issue": "Client API fails to resolve auth signature against external auth helper.",
        "file": "client.py",
        "code": (
            "def sign_request(payload: str) -> str:\n"
            "    return f'signed:{payload}'\n"
        ),
        "test_file": "test_client.py",
        "test_code": (
            "import pytest\n"
            "from client import sign_request\n"
            "def test_sign():\n"
            "    # External latent dependency not in local repo\n"
            "    raise ImportError('ModuleNotFoundError: No module named external_auth_provider (latent dependency across multi-file boundary)')\n"
        ),
        "model_responses": [
            (
                "<|think|>\n"
                "Updating request signature.\n"
                "</|think>\n"
                "<edit file=\"client.py\" start_line=\"1\" end_line=\"2\">\n"
                "<old>\n"
                "def sign_request(payload: str) -> str:\n"
                "    return f'signed:{payload}'\n"
                "</old>\n"
                "<new>\n"
                "def sign_request(payload: str) -> str:\n"
                "    return f'v2_signed:{payload}'\n"
                "</new>\n"
                "</edit>\n"
            )
        ],
    },
    # --------------------------------------------------------------------------
    # Case 7: Unresolved - Dynamic Reflection / Monkey-patch
    # --------------------------------------------------------------------------
    {
        "id": "unresolved_dynamic_reflection",
        "issue": "Dispatcher fails dynamic reflection resolution on unmapped actions.",
        "file": "dispatch.py",
        "code": (
            "class Dispatcher:\n"
            "    def dispatch(self, action: str):\n"
            "        handler = getattr(self, f'handle_{action}')\n"
            "        return handler()\n"
        ),
        "test_file": "test_dispatch.py",
        "test_code": (
            "from dispatch import Dispatcher\n"
            "def test_dispatch():\n"
            "    d = Dispatcher()\n"
            "    # Dynamic reflection / monkeypatch failure\n"
            "    raise AttributeError(\"'Dispatcher' object has no attribute 'handle_dynamic_eval' (reflection failure)\")\n"
        ),
        "model_responses": [
            (
                "<|think|>\n"
                "Adding fallback return.\n"
                "</|think>\n"
                "<edit file=\"dispatch.py\" start_line=\"2\" end_line=\"4\">\n"
                "<old>\n"
                "    def dispatch(self, action: str):\n"
                "        handler = getattr(self, f'handle_{action}')\n"
                "        return handler()\n"
                "</old>\n"
                "<new>\n"
                "    def dispatch(self, action: str):\n"
                "        return getattr(self, f'handle_{action}', lambda: None)()\n"
                "</new>\n"
                "</edit>\n"
            )
        ],
    },
    # --------------------------------------------------------------------------
    # Case 8: Unresolved - Underspecified Issue Text
    # --------------------------------------------------------------------------
    {
        "id": "unresolved_underspecified_issue",
        "issue": "underspecified issue: optimize internal state processing without altering any external invariants.",
        "file": "service.py",
        "code": (
            "class Service:\n"
            "    def process(self, x):\n"
            "        return x * 2\n"
        ),
        "test_file": "test_service.py",
        "test_code": (
            "from service import Service\n"
            "def test_proc():\n"
            "    s = Service()\n"
            "    assert s.process(5) == 10\n"
        ),
        "model_responses": [
            (
                "<|think|>\n"
                "The issue statement is underspecified. No target metrics, invariants, or requirements are given.\n"
                "I cannot safely generate edits without specifications.\n"
                "</|think>\n"
                "Clarification requested: the issue description is underspecified."
            )
        ],
    },
    # --------------------------------------------------------------------------
    # Case 9: Unresolved - Harness Timeout / Deadlock
    # --------------------------------------------------------------------------
    {
        "id": "unresolved_harness_timeout",
        "issue": "Worker thread encounters lock acquisition deadlock in test harness.",
        "file": "worker.py",
        "code": (
            "def run_worker():\n"
            "    return 'done'\n"
        ),
        "test_file": "test_worker.py",
        "test_code": (
            "def test_timeout():\n"
            "    # Simulated harness timeout / lock acquisition deadlock\n"
            "    assert False, 'Harness Timeout: worker process deadlock waiting for synchronizer lock'\n"
        ),
        "model_responses": [
            (
                "<|think|>\n"
                "Checking worker implementation.\n"
                "</|think>\n"
                "<edit file=\"worker.py\" start_line=\"1\" end_line=\"2\">\n"
                "<old>\n"
                "def run_worker():\n"
                "    return 'done'\n"
                "</old>\n"
                "<new>\n"
                "def run_worker():\n"
                "    return 'complete'\n"
                "</new>\n"
                "</edit>\n"
            )
        ],
    },
    # --------------------------------------------------------------------------
    # Case 10: Unresolved - Premature Search Termination
    # --------------------------------------------------------------------------
    {
        "id": "unresolved_premature_termination",
        "issue": "Fix cache eviction race condition in SessionManager.",
        "file": "cache.py",
        "code": (
            "class Cache:\n"
            "    def evict(self):\n"
            "        return False\n"
        ),
        "test_file": "test_cache.py",
        "test_code": (
            "from cache import Cache\n"
            "def test_cache():\n"
            "    c = Cache()\n"
            "    assert c.evict() is True\n"
        ),
        "model_responses": [
            (
                "<|think|>\n"
                "I analyzed the repository call graph but cannot locate any bug in Cache.evict.\n"
                "Terminating search prematurely.\n"
                "</|think>\n"
                "Analysis complete: no edits proposed."
            )
        ],
    },
]


def run_repair_eval(backend: Optional[ModelBackend] = None) -> List[TrajectoryResult]:
    """
    Run stratified end-to-end evaluation using PerronAgent across defect archetypes.
    """
    trajectories: List[TrajectoryResult] = []

    print("=" * 70)
    print("PERRON AUTONOMOUS DEVELOPER AGENT EVALUATION")
    print("=" * 70)

    for case in BENCHMARK_CASES:
        case_id = case["id"]
        print(f"\n[EVAL] Running Instance: {case_id}")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()

            # Write source and test files
            src_path = repo_path / case["file"]
            src_path.write_text(case["code"], encoding="utf-8")

            test_path = repo_path / case["test_file"]
            test_path.write_text(case["test_code"], encoding="utf-8")

            # Configure agent backend with list of canned responses
            canned = case.get("model_responses", [])
            case_backend = backend or OfflineReplayBackend(canned_responses=canned)

            with PerronAgent(
                repo_dir=repo_path,
                backend=case_backend,
                max_turns=3,
            ) as agent:
                agent.initialize_graph()

                result = agent.solve_issue(
                    issue_text=case["issue"],
                    test_file=str(test_path),
                    instance_id=case_id,
                )

            trajectories.append(result)

            status = "PASS" if result.resolved and result.test_passed else "FAIL"
            failure_info = f" (Category: {result.failure_category})" if not result.resolved else ""
            print(f"       Status: {status}{failure_info} | Duration: {result.total_duration_seconds:.2f}s | Turns: {result.total_turns}")
            if result.thinking_traces:
                first_trace = result.thinking_traces[0].replace("\n", " ")[:80]
                print(f"       Thinking Trace: {first_trace}...")

    # Aggregate telemetry and export
    telemetry_data = aggregate_trajectories(trajectories)
    save_telemetry(telemetry_data)
    print(f"\n[TELEMETRY] Exported {len(trajectories)} trajectories to benchmarks/run_telemetry.json")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    resolved_count = sum(1 for t in trajectories if t.resolved and t.test_passed)
    total_count = len(trajectories)
    print(f"Resolved: {resolved_count}/{total_count} ({resolved_count/total_count*100:.1f}%)")
    print(f"Failure Mode Distribution:")
    for label, pct in zip(telemetry_data["failure_labels"], telemetry_data["failure_percentages"]):
        print(f"  - {label}: {pct}%")
    print("=" * 70)

    return trajectories


if __name__ == "__main__":
    run_repair_eval()
