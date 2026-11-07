"""Build every split that is generated from the expert, with a fixed seed ledger.

Supersedes `data/splits/generate_splits.py`.  Two defects in that script are
corrected here:

1. **Seed leakage.** `drift_targeted` used seeds 2000-2019 and
   `drift_train/v0_failed_rollouts.jsonl` holds real model failures on those
   *same* seeds.  The healer was training on the exact tasks the gate grades —
   the one thing the plan says must never happen.  The buckets now use disjoint
   seed blocks and the overlap is asserted, not assumed.

2. **`drift_targeted` was not drifted.** It was generated as clean expert
   traces, so gate bucket 1 asked "did we fix the failure?" about tasks with no
   failure in them.  It is now generated under D1, the demo drift, on held-out
   seeds.

The sealed set is deliberately **not** regenerated: its sha256 is already
committed, it has not been opened, and rewriting it would silently move the
goalposts.  It is used as a task list, and its ground truth is unchanged because
the oracle did not change.

Usage:
    python -m data.splits.build_splits
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from expert.scripted import run_expert  # noqa: E402
from world.snapshot import generate_snapshot  # noqa: E402

# The seed ledger.  One block per split, never shared, never reused.
SEED_LEDGER: dict[str, tuple[int, int]] = {
    "train_clean": (0, 100),
    "dev": (1000, 20),
    # 2000-2019 is frozen: real v0 failures live there in drift_train.
    "final_sealed": (5000, 50),
    "drift_targeted": (6000, 60),
    "gate_regression": (6500, 60),
}

FROZEN = {"final_sealed"}

# Gate bucket 2 is clean tasks plus the other drifts, so a D1 fix that breaks D2
# cannot hide inside a good average.
GATE_REGRESSION_DRIFTS = (
    [(None, 42)] + [("D2", 6), ("D3", 6), ("D4", 6)]
)


def _write(path: Path, traces: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace) + "\n")


def build_train_clean() -> None:
    start, count = SEED_LEDGER["train_clean"]
    traces = []
    for offset in range(count):
        trace = run_expert(generate_snapshot(start + offset), inject_error=True)
        trace["split"] = "train_clean"
        traces.append(trace)
    _write(ROOT / "data" / "train_clean" / "expert_traces.jsonl", traces)
    print(f"train_clean: {count} traces (seeds {start}-{start + count - 1}, "
          f"with a changed retry after a 503)")


def build_dev() -> None:
    start, count = SEED_LEDGER["dev"]
    traces = []
    for offset in range(count):
        trace = run_expert(generate_snapshot(start + offset))
        trace["split"] = "dev"
        traces.append(trace)
    _write(ROOT / "data" / "dev" / "expert_traces.jsonl", traces)
    print(f"dev: {count} traces (seeds {start}-{start + count - 1})")


def build_drift_targeted() -> None:
    start, count = SEED_LEDGER["drift_targeted"]
    traces = []
    for offset in range(count):
        trace = run_expert(generate_snapshot(start + offset), drift="D1")
        trace["split"] = "drift_targeted"
        traces.append(trace)
    _write(ROOT / "data" / "drift_targeted" / "expert_traces.jsonl", traces)
    print(f"drift_targeted: {count} D1 traces (seeds {start}-{start + count - 1})")


def build_gate_regression() -> None:
    start, count = SEED_LEDGER["gate_regression"]
    traces = []
    seed = start
    for drift, block in GATE_REGRESSION_DRIFTS:
        for _ in range(block):
            trace = run_expert(generate_snapshot(seed), drift=drift)
            trace["split"] = "gate_regression"
            traces.append(trace)
            seed += 1
    if seed != start + count:
        raise AssertionError(f"gate_regression ledger mismatch: {seed - start} != {count}")
    _write(ROOT / "data" / "gate_regression" / "expert_traces.jsonl", traces)
    print(f"gate_regression: {count} traces (42 clean, 6 D2, 6 D3, 6 D4)")


def assert_no_leakage() -> None:
    seen: dict[int, str] = {}

    for name, (start, count) in SEED_LEDGER.items():
        for offset in range(count):
            seed = start + offset
            if seed in seen:
                raise AssertionError(f"seed {seed} is in both {seen[seed]} and {name}")
            seen[seed] = name

    rollouts = ROOT / "data" / "drift_train" / "v0_failed_rollouts.jsonl"
    if rollouts.is_file():
        with rollouts.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                seed = json.loads(line)["seed"]
                if seed in seen:
                    raise AssertionError(
                        f"drift_train seed {seed} collides with {seen[seed]}"
                    )
    print(f"leakage check: {len(seen)} seeds, all disjoint")


def sealed_hash_is_unchanged() -> bool:
    sealed = ROOT / "data" / "final_sealed" / "sealed_traces.jsonl"
    recorded = (ROOT / "data" / "splits" / "sealed_hash.txt").read_text().strip()
    content = sealed.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest() == recorded


def main() -> None:
    assert_no_leakage()
    build_train_clean()
    build_dev()
    build_drift_targeted()
    build_gate_regression()

    if not sealed_hash_is_unchanged():
        raise SystemExit("sealed hash check failed — the frozen set was modified")
    print("sealed set untouched, hash still matches data/splits/sealed_hash.txt")


if __name__ == "__main__":
    main()
