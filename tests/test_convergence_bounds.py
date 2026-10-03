import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_iteration_bound():
    assert math.ceil(math.log(2 / 1e-6) / math.log(1 / 0.85)) == 90


def test_walk_mass_constants():
    assert abs(0.85 ** 3 - 0.614125) < 1e-12
    assert abs(0.85 / 0.15 - 5.666666666666667) < 1e-12


def test_spectral_artifact_matches_bound():
    d = json.loads((ROOT / "results" / "spectral_convergence.json").read_text(encoding="utf-8"))
    assert d["theoretical_bounds"]["theoretical_max_iterations"] == 90
    for repo in ("requests", "sympy"):
        assert d["empirical_convergence"][repo]["power_iterations"] <= 90
        assert d["empirical_convergence"][repo]["final_l1_residual"] < 1e-6
