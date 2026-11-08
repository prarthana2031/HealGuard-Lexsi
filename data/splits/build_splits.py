"""Build every expert-generated split for world v2.

Two properties this file guarantees, and one it deliberately refuses to do.

Guarantees
----------
1. **Disjoint seeds.** Every split draws from its own pool. The pools used by
   the classifier battery (7000-7400) and the healer's simulator seeds
   (8000-8100) are reserved too. `data/splits/seed_ledger.json` records the exact
   seeds so a regeneration is auditable.
2. **Answer balance by construction.** The plan asks for about half of the tasks
   in *every* split to have the empty answer. v2's generator is naturally ~41%
   none, so each split selects seeds until it has exactly half `none` and half
   vulnerable, then interleaves them. `generate_snapshot(seed)` stays a pure
   function; only *which* seeds a split claims is chosen.

Refuses
-------
It does not resurrect `data/drift_train/v0_failed_rollouts.jsonl`. Those are real
model failures observed against **world v1**; under v2 the same seed is a
different account, so they are archived rather than silently reused.

The sealed set *is* regenerated in this pass. That is a deliberate pre-open
re-freeze, recorded in `data/splits/sealed_freeze.md`, because world v2 changed
every snapshot. It is legal only because the set has never been opened and no
published number depended on it.

Usage:
    python -m data.splits.build_splits
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from expert.scripted import run_expert  # noqa: E402
from oracle.checker import check_vulnerabilities  # noqa: E402
from world.snapshot import generate_snapshot  # noqa: E402

# pool start, pool size, tasks wanted
SCHEDULE: dict[str, tuple[int, int, int]] = {
    "train_clean": (0, 1500, 400), # <--- Changed from 40 to 400
    "dev": (3000, 1000, 40),
    "drift_targeted": (6000, 1000, 60),
    "gate_regression": (10000, 1000, 60),
    "final_sealed": (20000, 1000, 50),
}

# The drift for each bucket. `gate_regression` mixes clean with the other drifts
# so a D1 fix that breaks D2 cannot hide inside a good average.
GATE_REGRESSION_DRIFTS = [(None, 42), ("D2", 6), ("D3", 6), ("D4", 6)]

RESERVED = {
    "archive_world_v1": (2000, 100),
    "classifier_battery": (7000, 400),
    "heal_simulator": (8000, 100),
}

LEDGER = ROOT / "data" / "splits" / "seed_ledger.json"
HASH_FILE = ROOT / "data" / "splits" / "sealed_hash.txt"
FREEZE_FILE = ROOT / "data" / "splits" / "sealed_freeze.md"
HISTORY_FILE = ROOT / "data" / "splits" / "sealed_history.json"


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------

def select_balanced(pool: int, pool_size: int, count: int) -> list[tuple[int, str]]:
    """Pick exactly half `none` and half vulnerable seeds, then interleave.

    Returns `[(seed, kind)]` where kind is "none" or "vulnerable".
    """
    want_none = count // 2
    want_vulnerable = count - want_none

    none_seeds: list[int] = []
    vulnerable_seeds: list[int] = []

    for seed in range(pool, pool + pool_size):
        if len(none_seeds) >= want_none and len(vulnerable_seeds) >= want_vulnerable:
            break
        is_none = not check_vulnerabilities(generate_snapshot(seed))
        if is_none and len(none_seeds) < want_none:
            none_seeds.append(seed)
        elif not is_none and len(vulnerable_seeds) < want_vulnerable:
            vulnerable_seeds.append(seed)

    if len(none_seeds) < want_none or len(vulnerable_seeds) < want_vulnerable:
        raise RuntimeError(
            f"pool {pool}..{pool + pool_size} could not supply "
            f"{want_none} none / {want_vulnerable} vulnerable tasks "
            f"(got {len(none_seeds)} / {len(vulnerable_seeds)})"
        )

    ordered: list[tuple[int, str]] = []
    for index in range(count):
        if index % 2 == 0:
            ordered.append((none_seeds.pop(0), "none"))
        else:
            ordered.append((vulnerable_seeds.pop(0), "vulnerable"))
    return ordered


def assert_disjoint(ledger: dict[str, list[int]]) -> None:
    seen: dict[int, str] = {}

    for name, seeds in ledger.items():
        for seed in seeds:
            if seed in seen:
                raise AssertionError(f"seed {seed} in both {seen[seed]} and {name}")
            seen[seed] = name

    for name, (start, size) in RESERVED.items():
        for seed in range(start, start + size):
            if seed in seen:
                raise AssertionError(
                    f"reserved block {name} seed {seed} collides with {seen[seed]}"
                )

    archived = ROOT / "data" / "archive" / "world_v1" / "v0_failed_rollouts.jsonl"
    if archived.is_file():
        with archived.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                seed = json.loads(line)["seed"]
                if seed in seen:
                    raise AssertionError(
                        f"archived world-v1 seed {seed} collides with {seen[seed]}"
                    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write(relative: str, traces: list[dict]) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace) + "\n")


def build_split(
    name: str,
    assignments: list[tuple[int, str]],
    drift: str | None,
    filename: str = "expert_traces.jsonl",
) -> None:
    traces = []
    for i, (seed, _kind) in enumerate(assignments):
        # Inject error in 20% of traces, but ONLY for train_clean
        inject = (i % 5 == 0) if name == "train_clean" else False
        
        # Note: Added inject_error=inject to the run_expert call
        trace = run_expert(generate_snapshot(seed), inject_error=inject, drift=drift)
        
        trace["split"] = name
        trace["answer_kind"] = _kind
        traces.append(trace)
    _write(f"data/{name}/{filename}", traces)


def build_gate_regression(name: str, assignments: list[tuple[int, str]]) -> None:
    """42 clean + 6 D2 + 6 D3 + 6 D4, in the order the assignments were drawn."""
    drifts = [drift for drift, block in GATE_REGRESSION_DRIFTS for _ in range(block)]
    if len(drifts) != len(assignments):
        raise AssertionError("gate_regression drift plan does not match task count")

    traces = []
    for (seed, kind), drift in zip(assignments, drifts):
        trace = run_expert(generate_snapshot(seed), drift=drift)
        trace["split"] = name
        trace["answer_kind"] = kind
        traces.append(trace)
    _write(f"data/{name}/expert_traces.jsonl", traces)


def sealed_hash() -> str:
    sealed = ROOT / "data" / "final_sealed" / "sealed_traces.jsonl"
    return hashlib.sha256(sealed.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def record_freeze(new_hash: str) -> list[dict]:
    """Append the new hash to the freeze history, then render the record.

    Append-only on purpose: re-running the builder must not rewrite history.
    The first version this repo ever committed is kept as `world v1` even after
    the file it described has been replaced.
    """
    history: list[dict] = []
    if HISTORY_FILE.is_file():
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

    if not history:
        # Seed the history with the hash that was committed before world v2.
        history = [
            {
                "version": "world v1",
                "sha256": "fc7dcf2d675620db4aa12077ac9a8cef175b46ab7960965939a6864ace3a46a7",
                "recorded_at": "2026-11-05T00:00:00+00:00",
                "reason": "Initial sealed set, generated before the world redesign.",
            }
        ]

    if history[-1]["sha256"] != new_hash:
        history.append(
            {
                "version": "world v2",
                "sha256": new_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "reason": (
                    "World v2: generator changed (families decoupled, answer "
                    "sizes varied, positions randomised), so every snapshot for "
                    "every seed changed."
                ),
            }
        )

    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")

    rows = "\n".join(
        f"| {entry['version']} | `{entry['sha256']}` |" for entry in history
    )
    FREEZE_FILE.write_text(
        "\n".join(
            [
                "# Sealed set freeze record",
                "",
                "Append-only. Generated from `data/splits/sealed_history.json` by",
                "`python -m data.splits.build_splits`; do not hand-edit.",
                "",
                f"- **Current hash:** `{new_hash}`",
                f"- **Task count:** 50 (inside the plan's 40-60 window)",
                "- **Answer balance:** 25 `none`, 25 vulnerable, interleaved",
                "- **Opened before any re-freeze?** No. No score, no table and no",
                "  article number was produced from the world-v1 sealed set.",
                "",
                "| Version | sha256 of `sealed_traces.jsonl` |",
                "|---|---|",
                rows,
                "",
                "The v2 hash is frozen for the 14 Nov submission. Any later change",
                "invalidates the article's claim that the set was opened exactly once.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return history


def main() -> None:
    ledger: dict[str, list[int]] = {}
    assignments: dict[str, list[tuple[int, str]]] = {}

    for name, (pool, pool_size, count) in SCHEDULE.items():
        chosen = select_balanced(pool, pool_size, count)
        assignments[name] = chosen
        ledger[name] = [seed for seed, _kind in chosen]
        none_count = sum(1 for _seed, kind in chosen if kind == "none")
        seeds = [seed for seed, _kind in chosen]
        print(
            f"  {name:17s} n={count:3d} none={none_count:3d} "
            f"({none_count / count:.0%}) seeds {min(seeds)}-{max(seeds)}"
        )

    assert_disjoint(ledger)

    build_split("train_clean", assignments["train_clean"], drift=None)
    build_split("dev", assignments["dev"], drift=None)
    build_split("drift_targeted", assignments["drift_targeted"], drift="D1")
    build_gate_regression("gate_regression", assignments["gate_regression"])
    build_split(
        "final_sealed",
        assignments["final_sealed"],
        drift=None,
        filename="sealed_traces.jsonl",
    )

    new_hash = sealed_hash()
    HASH_FILE.write_text(new_hash, encoding="utf-8")
    record_freeze(new_hash)

    LEDGER.write_text(
        json.dumps(
            {
                "world": "v2",
                "generated_by": "data.splits.build_splits",
                "schedule": {name: list(value) for name, value in SCHEDULE.items()},
                "reserved": RESERVED,
                "seeds": ledger,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n  sealed hash: {new_hash}")
    print(f"  freeze record: {FREEZE_FILE.relative_to(ROOT)}")
    print(f"  ledger: {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
