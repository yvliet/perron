"""
Frozen verification test for data/swebench_lite_split.json.

Enforces:
1. Exact cryptographic SHA256 integrity (no modifications permitted).
2. Partition size guarantees: dev_pilot (50), dev_val (100), heldout (150).
3. Strict disjointness across all three partitions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FROZEN_SHA256_CRLF = "18a947a0e8a7941231ab15f11c1eee335adc58f8d03cbce742637452c9a23418"
FROZEN_SHA256_LF = "95c7ef3ca69214f5ac9f6bcba20e2531596ed260c2cbe181d2df3a56873d259b"
FROZEN_SHA256 = FROZEN_SHA256_CRLF
SPLIT_PATH = Path(__file__).resolve().parent.parent / "data" / "swebench_lite_split.json"


def test_swebench_lite_split_hash() -> None:
    """Assert data/swebench_lite_split.json has not been modified (accepting CRLF or LF checkout)."""
    assert SPLIT_PATH.is_file(), f"Split file not found at {SPLIT_PATH}"
    with open(SPLIT_PATH, "rb") as f:
        computed_sha256 = hashlib.sha256(f.read()).hexdigest()

    assert computed_sha256 in (FROZEN_SHA256_CRLF, FROZEN_SHA256_LF), (
        f"Cryptographic drift detected! data/swebench_lite_split.json SHA256 "
        f"expected one of {(FROZEN_SHA256_CRLF, FROZEN_SHA256_LF)}, got {computed_sha256}"
    )


def test_swebench_lite_split_partitions() -> None:
    """Assert partition sizes (50/100/150) and pairwise disjointness."""
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    pilot = data.get("dev_pilot_instances", [])
    val = data.get("dev_val_instances", [])
    heldout = data.get("heldout_instances", [])

    assert len(pilot) == 50, f"Expected 50 pilot instances, found {len(pilot)}"
    assert len(val) == 100, f"Expected 100 dev_val instances, found {len(val)}"
    assert len(heldout) == 150, f"Expected 150 heldout instances, found {len(heldout)}"

    set_pilot = set(pilot)
    set_val = set(val)
    set_heldout = set(heldout)

    assert len(set_pilot) == 50, "Duplicate instance IDs detected in dev_pilot"
    assert len(set_val) == 100, "Duplicate instance IDs detected in dev_val"
    assert len(set_heldout) == 150, "Duplicate instance IDs detected in heldout"

    pilot_val_intersect = set_pilot & set_val
    assert not pilot_val_intersect, f"dev_pilot and dev_val overlap on: {pilot_val_intersect}"

    pilot_heldout_intersect = set_pilot & set_heldout
    assert not pilot_heldout_intersect, f"dev_pilot and heldout overlap on: {pilot_heldout_intersect}"

    val_heldout_intersect = set_val & set_heldout
    assert not val_heldout_intersect, f"dev_val and heldout overlap on: {val_heldout_intersect}"
