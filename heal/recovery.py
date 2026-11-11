"""Build recovery-conditioned SFT traces from real v0 drift rollouts.

Why this module exists
----------------------
v1_sft was trained on `data/heal/sft_chosen.jsonl`, where the system prompt
advertises `bucket` and the assistant calls `bucket_name` with **no error turn in
between**.  The model could not learn *when* to switch names, so it learned a
mixture; on clean traffic the mixture leaks and becomes a schema violation.

This module builds the opposite object: a trace in which the stale call is
followed by the real error observation and then the corrected call.

    assistant  get_bucket_acl(bucket=...)
    tool       {"error": "... unexpected argument(s): bucket; use bucket_name"}
    assistant  get_bucket_acl(bucket_name=...)

That is a conditional correction: on the clean API the error never occurs, so
the branch is never taken.  The invariant is enforced here at build time and
again in `tests/test_recovery_data.py`.

Sources (all disjoint from the 40k fresh/gate block)
----------------------------------------------------
`data/drift_train/*.jsonl`
    successful v0 rollouts on seeds 30000-30059 / 50000-50099.
`data/heal/clean_mix.jsonl`
    clean traces on seeds 8000-8025.
`data/train_clean/expert_traces.jsonl`
    extra clean traces, used only to fill the clean quota.

Usage
-----
    python -m heal.recovery
    python -m heal.recovery --drift D1 --clean-ratio 0.5 --max-steps 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from expert.scripted import TEMPLATE_ID
from heal.drifts.schema_diff import SCHEMA_DIFFS, get_schema_diff
from heal.examples import evaluation_seeds
from heal.replay import parse_calls, replay
from world.snapshot import generate_snapshot

ROOT = Path(__file__).resolve().parents[1]
DRIFT_TRAIN_DIR = ROOT / "data" / "drift_train"
CLEAN_SOURCES = (
    ROOT / "data" / "heal" / "clean_mix.jsonl",
    ROOT / "data" / "train_clean" / "expert_traces.jsonl",
)
OUT_PATH = ROOT / "data" / "heal" / "recovery_sft.jsonl"

# Every argument/tool name any drift introduces.  A clean trace must not contain
# one of these: that is the exact defect that produced v1_sft's 38%.
ALL_DRIFTED_NAMES: set[str] = set()
for _diff in SCHEMA_DIFFS.values():
    ALL_DRIFTED_NAMES |= _diff.new_arg_names()
    ALL_DRIFTED_NAMES |= set(_diff.renamed_tools.values())


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------

def _tool_observation(messages: list[dict[str, Any]], index: int) -> str:
    """The tool observation immediately before assistant message `index`."""
    if index == 0:
        return ""
    previous = messages[index - 1]
    if previous.get("role") != "tool":
        return ""
    return str(previous.get("content", ""))


def _calls_in(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        call
        for call in parse_calls([message])
        if isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict)
    ]


def check_recovery_invariant(
    messages: list[dict[str, Any]],
    drift: str,
) -> list[str]:
    """Every drifted name must be preceded by an error that names it.

    Returns a list of human-readable violations; empty means the trace is a
    conditional correction, not a schema override.
    """

    diff = get_schema_diff(drift)
    if diff is None:
        return ["drift is required for a recovery trace"]

    new_args = diff.new_arg_names()
    new_tools = set(diff.renamed_tools.values())
    violations: list[str] = []

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        observation = _tool_observation(messages, index)

        for call in _calls_in(message):
            used_new_arg = set(call["arguments"]) & new_args
            used_new_tool = call["name"] in new_tools
            if not used_new_arg and not used_new_tool:
                continue

            named = used_new_arg | ({call["name"]} if used_new_tool else set())
            if "error" not in observation.lower():
                violations.append(
                    f"message {index}: {call['name']} uses {sorted(named)} "
                    "with no preceding error observation"
                )
                continue
            for name in named:
                if name not in observation:
                    violations.append(
                        f"message {index}: {call['name']} uses {name!r} but the "
                        f"preceding error does not name it: {observation[:160]!r}"
                    )

    return violations


def clean_violations(messages: list[dict[str, Any]]) -> list[str]:
    """A clean trace must not contain any drifted name at all."""

    violations: list[str] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in _calls_in(message):
            used = set(call["arguments"]) & ALL_DRIFTED_NAMES
            if call["name"] in ALL_DRIFTED_NAMES:
                used.add(call["name"])
            if used:
                violations.append(
                    f"message {index}: clean trace uses drifted name(s) {sorted(used)}"
                )
    return violations


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_successful_drift_rows(
    drift: str,
    reserved: set[int],
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One shortest successful trace per training seed, leak-checked."""

    stats = {"read": 0, "leaked": 0, "failed": 0, "too_long": 0}
    by_seed: dict[int, dict[str, Any]] = {}

    for path in sorted(DRIFT_TRAIN_DIR.glob("*.jsonl")):
        for row in _read_jsonl(path):
            if row.get("drift") != drift:
                continue
            stats["read"] += 1
            seed = row.get("seed")
            if seed is None:
                continue
            if seed in reserved:
                stats["leaked"] += 1
                continue
            if not row.get("success"):
                stats["failed"] += 1
                continue
            if not row.get("messages"):
                stats["failed"] += 1
                continue

            calls = row.get("tool_calls") or parse_calls(row["messages"])
            steps = len([call for call in calls if call.get("name") != "finish"])
            if steps > max_steps:
                stats["too_long"] += 1
                continue

            previous = by_seed.get(seed)
            if previous is None or len(row["messages"]) < len(previous["messages"]):
                row["_source_file"] = path.name
                by_seed[seed] = row

    return list(by_seed.values()), stats


def load_clean_rows(
    reserved: set[int],
    quota: int,
) -> list[dict[str, Any]]:
    """Verified clean traces, one per seed, until the quota is met."""

    rows: list[dict[str, Any]] = []
    seen: set[int] = set()

    for path in CLEAN_SOURCES:
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            if len(rows) >= quota:
                return rows
            seed = row.get("seed")
            if seed is None or seed in seen or seed in reserved:
                continue
            messages = row.get("messages")
            if not messages:
                continue
            if clean_violations(messages):
                continue

            calls = row.get("tool_calls") or parse_calls(messages)
            verified = replay(generate_snapshot(seed), calls, None)
            if not verified["correct"]:
                continue

            seen.add(seed)
            row["_source_file"] = path.name
            row["_replay"] = verified
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def build_recovery_row(
    row: dict[str, Any],
    drift: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Verify one rollout and convert it to an SFT row, or explain the skip."""

    seed = row["seed"]
    messages = row.get("messages") or []
    calls = row.get("tool_calls") or parse_calls(messages)

    violations = check_recovery_invariant(messages, drift)
    if violations:
        return None, f"invariant: {violations[0]}"

    verified = replay(generate_snapshot(seed), calls, drift)
    if not verified["correct"]:
        return None, "replay: final answer does not match the oracle"

    # Do not teach the flailing.  A trace that repeats an identical call or
    # loops is a success, but it is not the behaviour we want the adapter to
    # imitate: the gate also checks that the redundant-call rate does not rise.
    if verified["repeated_call"] or verified["loop"]:
        return None, "replay: repeated or looping call"

    error_count = sum(
        1
        for message in messages
        if message.get("role") == "tool"
        and "error" in str(message.get("content", "")).lower()
    )
    if error_count == 0:
        # A success with no error is a fine trace, but it is not a *recovery*
        # trace, and this file's whole point is the recovery turn.
        return None, "no error observation: not a recovery trace"

    calls_without_finish = [call for call in calls if call.get("name") != "finish"]
    out = {
        "id": f"rec_{drift}_{seed}",
        "seed": seed,
        "split": "heal_recovery",
        "drift": drift,
        "task": row.get("task", "Find all vulnerable cloud resources."),
        "template_id": row.get("template_id", TEMPLATE_ID),
        "messages": messages,
        "provenance": {
            "source": f"drift_train:{row['_source_file']}",
            "seed": seed,
            "drift": drift,
            "error_observations": error_count,
            "tool_calls": len(calls_without_finish),
            "replay_correct": verified["correct"],
            "replay_reward": verified["reward"],
            "repeated_call": verified["repeated_call"],
            "loop": verified["loop"],
            "invariant": "pass",
            "replay_hash": hashlib.sha256(
                json.dumps(calls, sort_keys=True).encode()
            ).hexdigest()[:16],
        },
        "success": True,
        "is_correct": True,
    }
    return out, None


def build(
    drift: str = "D1",
    clean_ratio: float = 0.5,
    max_steps: int = 24,
) -> dict[str, Any]:
    reserved = evaluation_seeds()
    recovery_rows, load_stats = load_successful_drift_rows(drift, reserved, max_steps)

    built: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for row in recovery_rows:
        out, reason = build_recovery_row(row, drift)
        if out is None:
            key = (reason or "unknown").split(":")[0]
            skipped[key] = skipped.get(key, 0) + 1
            continue
        built.append(out)

    if not built:
        raise SystemExit(
            "no recovery rows could be built. Check that data/drift_train has "
            "successful D1 rollouts on non-evaluation seeds."
        )

    # Clean quota is a share of the final mix, not of the recovery half.
    clean_quota = max(1, round(len(built) * clean_ratio / (1 - clean_ratio)))
    clean_rows = load_clean_rows(reserved, clean_quota)

    mix = built + [
        {
            "id": f"rec_clean_{row['seed']}",
            "seed": row["seed"],
            "split": "heal_recovery",
            "drift": None,
            "task": row.get("task", "Find all vulnerable cloud resources."),
            "template_id": row.get("template_id", TEMPLATE_ID),
            "messages": row["messages"],
            "provenance": {
                "source": f"clean:{row['_source_file']}",
                "seed": row["seed"],
                "drift": None,
                "error_observations": 0,
                "tool_calls": len(
                    [c for c in (row.get("tool_calls") or parse_calls(row["messages"]))
                     if c.get("name") != "finish"]
                ),
                "replay_correct": True,
                "replay_reward": row["_replay"]["reward"],
                "invariant": "clean",
                "replay_hash": hashlib.sha256(
                    json.dumps(row.get("tool_calls") or [], sort_keys=True).encode()
                ).hexdigest()[:16],
            },
            "success": True,
            "is_correct": True,
        }
        for row in clean_rows
    ]

    # Deterministic order: recovery first, clean second, seeds ascending.
    mix.sort(key=lambda row: (row["drift"] is None, row["seed"]))

    manifest = {
        "generated_by": "heal.recovery",
        "drift": drift,
        "template_id": TEMPLATE_ID,
        "clean_ratio_target": clean_ratio,
        "max_steps": max_steps,
        "reserved_evaluation_seeds": len(reserved),
        "drift_rows_read": load_stats["read"],
        "leaked_rows_excluded": load_stats["leaked"],
        "failed_rows": load_stats["failed"],
        "too_long_rows": load_stats["too_long"],
        "recovery_rows": len(built),
        "clean_rows": len(clean_rows),
        "clean_share": round(
            len(clean_rows) / (len(built) + len(clean_rows)), 4
        ),
        "total_rows": len(mix),
        "seeds": sorted(row["seed"] for row in mix),
        "skipped": skipped,
        "all_invariants_pass": all(
            check_recovery_invariant(row["messages"], row["drift"]) == []
            for row in built
        ),
        "all_replays_correct": all(
            row["provenance"].get("replay_correct") for row in mix
        ),
    }
    return {"rows": mix, "manifest": manifest}


def write(result: dict[str, Any], out_path: Path = OUT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        for row in result["rows"]:
            file.write(json.dumps(row) + "\n")
    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(result["manifest"], indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", default="D1", choices=sorted(SCHEMA_DIFFS))
    parser.add_argument("--clean-ratio", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    result = build(
        drift=args.drift,
        clean_ratio=args.clean_ratio,
        max_steps=args.max_steps,
    )
    out_path = Path(args.out)
    write(result, out_path)
    print(json.dumps(result["manifest"], indent=2))
    print(f"\nwrote {len(result['rows'])} rows to {out_path}")


if __name__ == "__main__":
    main()
