import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from perron.diffusion import personalized_pagerank_power_iteration

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "results" / "dev_val_beta_sweep.json"


def test_beta_zero_returns_teleport_prior():
    t = sp.csr_matrix(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
    p0 = np.array([0.5, 0.3, 0.2])
    pi = personalized_pagerank_power_iteration(t, np.zeros(3), p_0=p0, beta=0.0)
    np.testing.assert_allclose(pi, p0, atol=1e-12)


@pytest.mark.skipif(not DEV.exists(), reason="dev sweep artifact not generated")
def test_identity_methods_match_prior_on_dev():
    d = json.loads(DEV.read_text(encoding="utf-8"))
    ref = d["reference"]["prior_only"]["fn_acc_10"]["k"]
    for m, chk in d["identity_check"].items():
        assert chk["top10_set_equal_tasks"] == d["num_instances"], m
        assert d["sweep"][m]["beta_0.00"]["fn_acc_10"]["k"] == ref, m


@pytest.mark.skipif(not DEV.exists(), reason="dev sweep artifact not generated")
def test_selection_rule_is_reproducible():
    d = json.loads(DEV.read_text(encoding="utf-8"))
    for m, cells in d["sweep"].items():
        best = max(d["betas"], key=lambda b: (cells[f"beta_{b:.2f}"]["fn_acc_10"]["k"],
                                              cells[f"beta_{b:.2f}"]["fn_acc_5"]["k"], b))
        assert d["beta_star"][m] == best, m
