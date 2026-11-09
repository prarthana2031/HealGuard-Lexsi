"""Score every condition on the sealed set. One command, one table.

The sealed set is opened once (12 Nov). This module is the only thing that reads
it, it verifies the committed sha256 before and after, and it writes one JSON file
that `run_all.py --stage report` renders. No number in the article is typed by
hand.

    # everything (five conditions, ~200 model episodes)
    python -m eval.sealed_run

    # the headline first, if the clock is short (~2h)
    python -m eval.sealed_run --conditions expert,v0_d1_stale,v1_good_d1_stale

    # smoke the plumbing without a model
    python -m eval.sealed_run --conditions expert,v0_d1_stale --limit 3

Conditions
----------
`expert`            scripted expert, no model. The ceiling, and the number of
                    calls the task actually needs.
`untuned_clean`     base model, no adapter. The floor before any training.
`v0_clean`          v0 under the clean API. What the shipped model can do.
`v0_d1_stale`       v0 under D1 with a stale registry. The damage.
`v0_d1_prompt_only` v0, same drift, one sentence naming `bucket_name`.
`v1_good_d1_stale`  the healed adapter under the same drift.

Decoding is greedy, so the run is deterministic: this is one "seed", and the
article should say so rather than implying a second.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.splits.sealed import (  # noqa: E402
    load_sealed_tasks,
    sealed_hash,
    verify_sealed_hash,
)
from eval.rollout import PROMPT_VARIANTS, Backend, run_episode, to_record  # noqa: E402
from expert.scripted import run_expert  # noqa: E402
from heal.replay import replay  # noqa: E402
from world.snapshot import generate_snapshot  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "sealed"
RESULTS = OUT_DIR / "sealed_results.json"

V0 = "adapters/v0_answer2/final"
V1 = "adapters/v1_good/final"

# name -> (adapter or None, drift, stale_schema, prompt suffix)
CONDITIONS: dict[str, dict[str, Any]] = {
    "expert": {"kind": "expert"},
    "untuned_clean": {"adapter": None, "drift": None, "stale": False},
    "v0_clean": {"adapter": V0, "drift": None, "stale": False},
    "v0_d1_stale": {"adapter": V0, "drift": "D1", "stale": True},
    "v0_d1_prompt_only": {
        "adapter": V0,
        "drift": "D1",
        "stale": True,
        "prompt": PROMPT_VARIANTS["prompt_only_d1"],
    },
    "v1_good_d1_stale": {"adapter": V1, "drift": "D1", "stale": True},
}

DEFAULT_ORDER = list(CONDITIONS)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion. Reported because n is 50, not 500."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarise(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    none = [row for row in records if row.get("is_none_task")]
    vulnerable = [row for row in records if row.get("is_none_task") is False]
    successes = [row for row in records if row["success"]]
    steps = [row["num_steps"] for row in successes]
    calls = sum(row["total_tool_calls"] for row in records)
    redundant = sum(row["redundant_calls"] for row in records)

    def rate(rows: list[dict[str, Any]]) -> float | None:
        return sum(1 for row in rows if row["success"]) / len(rows) if rows else None

    none_rate = rate(none)
    vulnerable_rate = rate(vulnerable)
    low, high = wilson(len(successes), len(records))

    return {
        "condition": name,
        "n": len(records),
        "task_success": len(successes) / len(records) if records else 0.0,
        "ci95": [low, high],
        "success_on_none": none_rate,
        "success_on_vulnerable": vulnerable_rate,
        "balanced_success": (
            (none_rate + vulnerable_rate) / 2
            if none_rate is not None and vulnerable_rate is not None else None
        ),
        "median_steps": statistics.median(steps) if steps else None,
        "redundant_rate": redundant / calls if calls else 0.0,
        "mean_calls": calls / len(records) if records else 0.0,
    }


def expert_records(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ceiling. Deterministic, no model, no GPU."""
    records = []
    for task in tasks:
        snapshot = generate_snapshot(task["seed"])
        trace = run_expert(snapshot, drift=task["drift"])
        verified = replay(snapshot, trace["tool_calls"], task["drift"])
        records.append(
            {
                "id": trace["id"],
                "seed": task["seed"],
                "success": verified["correct"],
                "num_steps": trace["num_steps"],
                "total_tool_calls": trace["total_tool_calls"],
                "redundant_calls": trace["redundant_calls"],
                "is_none_task": not task["truth"],
            }
        )
    return records


def evaluate(
    backend_for: Callable[[str | None], Backend],
    tasks: list[dict[str, Any]],
    conditions: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run every condition on the same tasks and return the table."""
    selected = conditions or DEFAULT_ORDER
    chosen = tasks[:limit] if limit else tasks

    results: dict[str, Any] = {}
    # Group by adapter so a 3.35B checkpoint is loaded once, not once per condition.
    by_adapter: dict[str | None, list[str]] = {}
    for name in selected:
        spec = CONDITIONS[name]
        if spec.get("kind") == "expert":
            results[name] = summarise(name, expert_records(chosen))
            continue
        by_adapter.setdefault(spec["adapter"], []).append(name)

    for adapter, names in by_adapter.items():
        backend = backend_for(adapter)
        for name in names:
            spec = CONDITIONS[name]
            records = []
            for task in chosen:
                trajectory = run_episode(
                    backend,
                    task["seed"],
                    task["drift"],
                    prompt_suffix=spec.get("prompt", ""),
                    drift_override=spec["drift"],
                    stale_schema=spec["stale"],
                    max_steps=24,
                )
                records.append(
                    to_record(
                        trajectory,
                        bucket="final_sealed",
                        tag=name,
                        variant="clean",
                    )
                )
            results[name] = summarise(name, records)
            print(
                f"  {name:20s} n={len(records)} "
                f"success={results[name]['task_success']:.3f}",
                flush=True,
            )

    return results


def markdown(
    results: dict[str, Any], sealed_sha: str, none_tasks: int, total: int
) -> str:
    order = [name for name in DEFAULT_ORDER if name in results]
    lines = [
        "# Sealed set results",
        "",
        "Generated by `python -m eval.sealed_run`. Do not hand-edit.",
        "",
        f"- Sealed sha256: `{sealed_sha}`",
        f"- Tasks: {total} ({none_tasks} `none`, {total - none_tasks} vulnerable)",
        "- Decoding: greedy. Deterministic, so this is **one seed**.",
        "",
        "| Condition | N | Task success | 95% CI | On `none` | On vulnerable "
        "| Median steps | Redundant rate |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for name in order:
        row = results[name]
        low, high = row["ci95"]
        steps = f"{row['median_steps']:.1f}" if row["median_steps"] is not None else "n/a"
        none = f"{row['success_on_none']:.3f}" if row["success_on_none"] is not None else "n/a"
        vuln = (
            f"{row['success_on_vulnerable']:.3f}"
            if row["success_on_vulnerable"] is not None else "n/a"
        )
        lines.append(
            f"| {name} | {row['n']} | {row['task_success']:.3f} | "
            f"[{low:.2f}, {high:.2f}] | {none} | {vuln} | {steps} | "
            f"{row['redundant_rate']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the sealed set once.")
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated subset of: " + ", ".join(DEFAULT_ORDER),
    )
    parser.add_argument("--limit", type=int, help="First N sealed tasks (smoke).")
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--out", default=str(RESULTS), help="Where to write the JSON."
    )
    args = parser.parse_args()

    if not verify_sealed_hash():
        raise SystemExit(
            "sealed hash does not match data/splits/sealed_hash.txt - do not score it"
        )
    sha_before = sealed_hash()

    tasks = load_sealed_tasks()
    conditions = (
        DEFAULT_ORDER if args.conditions == "all" else args.conditions.split(",")
    )
    unknown = [name for name in conditions if name not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s): {unknown}")
    if args.limit:
        tasks = tasks[: args.limit]

    none_tasks = sum(1 for task in tasks if not task["truth"])
    print(
        f"sealed tasks: {len(tasks)} "
        f"({none_tasks} none, {len(tasks) - none_tasks} vulnerable)"
        f" | conditions: {', '.join(conditions)}"
    )
    print(f"sealed sha256: {sha_before}")

    def backend_for(adapter: str | None) -> Backend:
        from eval.rollout import TransformersBackend

        return TransformersBackend(
            adapter=str(ROOT / adapter) if adapter else None,
            model_id=args.model,
            temperature=args.temperature,
        )

    results = evaluate(backend_for, tasks, conditions, limit=None)

    sha_after = sealed_hash()
    if sha_after != sha_before:
        raise SystemExit("the sealed set changed during the run; results are void")

    payload = {
        "sealed_sha256": sha_before,
        "tasks": len(tasks),
        # A smoke run must never be mistaken for the sealed result. The report
        # shows a banner when this is false.
        "complete": len(tasks) == 50,
        "none_tasks": none_tasks,
        "decoding": "greedy" if args.temperature == 0 else f"temperature {args.temperature}",
        "conditions": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table = markdown(results, sha_before, none_tasks, len(tasks))
    (out.parent / "sealed_results.md").write_text(table, encoding="utf-8")
    print()
    print(table)
    print(f"wrote {out.relative_to(ROOT)} and sealed_results.md")


if __name__ == "__main__":
    main()
