#!/usr/bin/env python3
"""Two-stage freeze and verification for gemma4_protocol_v1.yaml.

Usage:
  python scripts/freeze_protocol.py stage1 benchmarks/protocol/gemma4_protocol_v1.yaml
  python scripts/freeze_protocol.py stage2 benchmarks/protocol/gemma4_protocol_v1.yaml benchmarks/protocol/gemma4_lock_v1.json
  python scripts/freeze_protocol.py verify benchmarks/protocol/gemma4_protocol_v1.yaml [benchmarks/protocol/gemma4_lock_v1.json]

Author: Sultan Haikal (GitHub: @yvliet)
License: Apache-2.0
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess
import sys

import yaml

RESOLVE = re.compile(r"<<RESOLVE_BEFORE_STAGE1:[^>]*>>")
LOCKPH = re.compile(r"<<LOCK_AT_STAGE2:[^>]*>>")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def die(msg):
    print(f"FREEZE ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_yaml(path):
    raw = path.read_bytes()
    if b"\r\n" in raw:
        die(f"{path} contains CRLF line endings; convert to LF before hashing")
    return raw, yaml.safe_load(raw.decode("utf-8"))


def record_path(path, stage):
    return path.with_suffix(path.suffix + f".stage{stage}.json")


def stage1(path, allow_dirty):
    raw, obj = load_yaml(path)
    left = RESOLVE.findall(raw.decode("utf-8"))
    if left:
        die(f"{len(left)} unresolved placeholder(s):\n  " + "\n  ".join(left))
    if git("rev-parse", "--is-inside-work-tree") == "true":
        if git("status", "--porcelain", "--", str(path)) and not allow_dirty:
            die("file has uncommitted changes; commit it first (or pass --allow-dirty)")
    else:
        print("warning: not inside a git repo; commit hash will be null", file=sys.stderr)
    rec = {
        "file": path.name,
        "raw_sha256": sha256(raw),
        "canonical_json_sha256": sha256(canon(obj)),
        "git_commit": git("rev-parse", "HEAD"),
        "frozen_utc": utcnow(),
        "stage": 1,
    }
    out = record_path(path, 1)
    out.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(rec, indent=2))
    print(f"\nwrote {out}\nnext: git add {path} {out} && git commit && "
          f"git tag -a protocol-gemma4-v1-stage1 -m {rec['raw_sha256']}")


def stage2(path, lock_path):
    raw, obj = load_yaml(path)
    rec1_path = record_path(path, 1)
    if not rec1_path.exists():
        die(f"{rec1_path} not found; run stage1 first")
    rec1 = json.loads(rec1_path.read_text(encoding="utf-8"))
    if sha256(raw) != rec1["raw_sha256"]:
        die("protocol file no longer matches its stage-1 hash")
    lock_raw = lock_path.read_bytes()
    lock = json.loads(lock_raw.decode("utf-8"))
    required = obj["lock"]["required_fields"]
    allowed = set(obj["lock"]["amendable_only_by_adding_these_fields"])
    missing = [f for f in required if f not in lock]
    if missing:
        die(f"lock is missing required fields: {missing}")
    extra = [k for k in lock if k not in allowed]
    if extra:
        die(f"lock contains fields not permitted by the protocol: {extra}")
    if LOCKPH.search(lock_raw.decode("utf-8")):
        die("lock still contains <<LOCK_AT_STAGE2>> placeholders")
    if lock["stage1_sha256_raw"] != rec1["raw_sha256"]:
        die("lock.stage1_sha256_raw does not match the stage-1 hash")
    if not all(lock["gates_passed"].get(g) is True for g in ("G0", "G1", "G2", "G3", "G4")):
        die("gates G0-G4 must all be true in lock.gates_passed")
    rec = {
        "file": lock_path.name,
        "raw_sha256": sha256(lock_raw),
        "canonical_json_sha256": sha256(canon(lock)),
        "stage1_raw_sha256": rec1["raw_sha256"],
        "git_commit": git("rev-parse", "HEAD"),
        "frozen_utc": utcnow(),
        "stage": 2,
    }
    out = record_path(lock_path, 2)
    out.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(rec, indent=2))
    print(f"\nwrote {out}\nnext: commit, tag protocol-gemma4-v1-stage2, THEN run Verified once.")


def verify(path, lock_path):
    raw, obj = load_yaml(path)
    rec1 = json.loads(record_path(path, 1).read_text(encoding="utf-8"))
    ok = sha256(raw) == rec1["raw_sha256"] and sha256(canon(obj)) == rec1["canonical_json_sha256"]
    print(f"[stage1] {path.name} raw_sha256={sha256(raw)} match={ok}")
    if lock_path:
        lraw = lock_path.read_bytes()
        rec2 = json.loads(record_path(lock_path, 2).read_text(encoding="utf-8"))
        ok2 = sha256(lraw) == rec2["raw_sha256"] and rec2["stage1_raw_sha256"] == rec1["raw_sha256"]
        print(f"[stage2] {lock_path.name} raw_sha256={sha256(lraw)} match={ok2}")
        ok = ok and ok2
    if not ok:
        die("HASH MISMATCH - do not run")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("stage1")
    s1.add_argument("protocol", type=pathlib.Path)
    s1.add_argument("--allow-dirty", action="store_true")
    s2 = sub.add_parser("stage2")
    s2.add_argument("protocol", type=pathlib.Path)
    s2.add_argument("lock", type=pathlib.Path)
    v = sub.add_parser("verify")
    v.add_argument("protocol", type=pathlib.Path)
    v.add_argument("lock", type=pathlib.Path, nargs="?")
    a = ap.parse_args()
    if a.cmd == "stage1":
        stage1(a.protocol, a.allow_dirty)
    elif a.cmd == "stage2":
        stage2(a.protocol, a.lock)
    else:
        verify(a.protocol, a.lock)


if __name__ == "__main__":
    main()
