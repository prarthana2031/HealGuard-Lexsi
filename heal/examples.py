"""Build corrective examples that replay to reward 1.

This is the Week 2 object the plan calls "the project":

* every failed drifted task is re-run with the scripted expert under the **new**
  schema, and the trace is kept only if replaying it in the environment returns
  reward 1;
* chosen and rejected share the prefix up to the first bad step, so DPO cannot
  learn "write longer" instead of "pass bucket_name";
* quotas per label, a cap per template, dedupe on
  sha256(template, error step, label), and provenance on every row;
* 30-50% clean traces are mixed back in so the heal does not wipe the clean
  skill.

Sources of failure:

`--source simulator` (default, no GPU)
    A deterministic stand-in for v0: it is given the **new** tool schema in its
    prompt and still emits the **old** argument names, which is exactly the
    failure a stale LoRA produces.  Every tool result is real, because the calls
    really run against the drifted environment.  Use this to validate the whole
    pipeline before a checkpoint exists.

`--source rollouts`
    `data/drift_train/v0_failed_rollouts.jsonl`, real model output.  If those
    traces contain no tool calls (untuned model) no minimal-edit pair can be
    built, and the tool says so instead of inventing one.

Usage:
    python -m heal.examples --drift D1
    python -m heal.examples --drift D1 --source rollouts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from classifier.rules import classify  # noqa: E402
from expert.scripted import (  # noqa: E402
    TEMPLATE_ID,
    TraceBuilder,
    render_call,
    run_expert,
    system_prompt_for,
    family_success,
)
from heal.drifts.registry import build_runtime, get_tool_schemas  # noqa: E402
from heal.replay import replay, replay_trajectory  # noqa: E402
from oracle.checker import check_vulnerabilities  # noqa: E402
from world.snapshot import generate_snapshot  # noqa: E402
from world.tools import CloudEnv  # noqa: E402

OUT_DIR = ROOT / "data" / "heal"
MIN_PER_LABEL = 8
MAX_PER_TEMPLATE = 40
CLEAN_MIX_MIN = 0.30
CLEAN_MIX_MAX = 0.50

# Seeds for the healer's own examples.  Deliberately away from every split.
SIM_SEEDS = range(8000, 8100)


# ---------------------------------------------------------------------------
# Failure sources
# ---------------------------------------------------------------------------

def stale_agent_trace(seed: int, drift: str) -> dict[str, Any]:
    """The same stale v0 policy the gate baseline uses.

    Shared on purpose: the examples must repair the failures the gate measures,
    not a different agent that happens to fail nearby.
    """
    from eval.run_agent import run_stale

    trace = run_stale(seed, drift)
    trace.update(
        {
            "split": "drift_train",
            "source": "simulator",
            "num_steps": len(
                [call for call in trace["tool_calls"] if call["name"] != "finish"]
            ),
            "total_tool_calls": len(trace["tool_calls"]),
            "valid_tool_calls": 0,
            "redundant_calls": 0,
            "reward": 0.0,
            "success": False,
            "is_correct": False,
            "finished": True,
            "error": None,
        }
    )
    return trace


def load_rollouts(drift: str) -> list[dict[str, Any]]:
    """Every `data/drift_train/*.jsonl` row for this drift.

    World v1 rollouts live in `data/archive/world_v1/` and are deliberately not
    read here: under world v2 the same seed is a different account, so a v1
    failure cannot be repaired into a v2 example.
    """
    rows = []
    for path in sorted((ROOT / "data" / "drift_train").glob("*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("drift") != drift:
                    continue
                row["source"] = f"rollouts:{path.name}"
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def _render_side(
    seed: int,
    drift: str,
    calls: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Canonical messages for a call sequence, executing each call for real.

    The pair is rebuilt from *calls*, never from the model's raw text.  A real
    rollout contains whatever whitespace and prose the model happened to emit,
    so requiring the failure's messages to be byte-identical to the expert's
    (the previous approach) rejected essentially every real pair.  The lesson we
    want the adapter to learn is the action at the first bad step, not the
    model's spacing.
    """
    snapshot = generate_snapshot(seed)
    env, schemas = build_runtime(CloudEnv(snapshot), drift=drift)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt_for(schemas)},
        {"role": "user", "content": "Find all vulnerable cloud resources."},
    ]

    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments", {})
        messages.append(
            {"role": "assistant", "content": render_call(name, arguments)}
        )

        if name == "finish":
            answer = arguments.get("answer") if isinstance(arguments, dict) else []
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "status": "success",
                            "answer": answer if isinstance(answer, list) else [],
                        },
                        default=str,
                    ),
                }
            )
            continue

        function = getattr(env, name, None)
        if callable(function):
            try:
                result = function(**arguments)
            except TypeError as exc:
                result = {"error": f"schema_violation: {exc}"}
        else:
            result = {"error": f"unknown_tool: Tool {name!r} not found"}

        messages.append({"role": "tool", "content": json.dumps(result, default=str)})

    return messages


def build_pair(failure: dict[str, Any], drift: str) -> dict[str, Any] | None:
    """A minimal-edit pair sharing every message up to the first bad step.

    Both sides are rendered canonically from their call sequences, so the shared
    prefix is guaranteed rather than hoped for, and the pair differs only at the
    first step where the two sequences disagree.
    """

    # A prompt-hinted run starts from a different system prompt than the expert's
    # canonical one, so a pair built from it would silently mix conditions. The
    # prompt-only arm is a control, not training data.
    if failure.get("prompt_variant", "clean") != "clean":
        return None

    seed = failure["seed"]
    snapshot = generate_snapshot(seed)

    chosen_trace = run_expert(snapshot, drift=drift)
    chosen_replay = replay(snapshot, chosen_trace["tool_calls"], drift)

    failure_calls = failure.get("tool_calls") or []
    chosen_calls = chosen_trace["tool_calls"]

    if not failure_calls:
        return None

    # First step where the two call sequences disagree.
    divergence = None
    for index in range(min(len(failure_calls), len(chosen_calls))):
        if failure_calls[index] != chosen_calls[index]:
            divergence = index
            break

    if divergence is None:
        return None

    chosen_messages = _render_side(seed, drift, chosen_calls)
    rejected_messages = _render_side(seed, drift, failure_calls)

    # Two messages per call (assistant + tool), plus system and user.
    prefix_length = 2 + 2 * divergence
    if chosen_messages[:prefix_length] != rejected_messages[:prefix_length]:
        return None  # the runs were not comparable; never force a pair

    rejected_replay = replay(snapshot, failure_calls, drift)

    if not chosen_replay["perfect"]:
        return None
    if rejected_replay["correct"]:
        return None

    classification = classify(failure)

    return {
        "prompt": chosen_messages[:prefix_length],
        "chosen": chosen_messages[prefix_length:],
        "rejected": rejected_messages[prefix_length:],
        "provenance": {
            "source_run": failure.get("task_id"),
            "source": failure.get("source"),
            "label": classification["primary"],
            "labels": classification["labels"],
            "drift": drift,
            "seed": seed,
            "divergence_step": divergence,
            "template_id": failure.get("template_id", TEMPLATE_ID),
            "prompt_variant": failure.get("prompt_variant", "clean"),
            "replay_hash": hashlib.sha256(
                json.dumps(chosen_trace["tool_calls"], sort_keys=True).encode()
            ).hexdigest()[:16],
            "chosen_reward": chosen_replay["reward"],
            "chosen_perfect": chosen_replay["perfect"],
            "rejected_reward": rejected_replay["reward"],
        },
    }


def _dedupe_key(pair: dict[str, Any]) -> str:
    """Dedupe a cascade, not every task.

    The plan says dedupe on (template, error step, label).  Applied without the
    snapshot that collapses the whole D1 bucket to a single row, because every
    D1 failure diverges at the same step with the same label.  Adding the
    snapshot keeps distinct held-out tasks while still refusing two rows for the
    same task.
    """
    provenance = pair["provenance"]
    return hashlib.sha256(
        f"{provenance['template_id']}|{provenance['divergence_step']}"
        f"|{provenance['label']}|{provenance['seed']}".encode()
    ).hexdigest()


def build(drift: str, source: str) -> dict[str, Any]:
    if source == "rollouts":
        failures = load_rollouts(drift)
    else:
        failures = [stale_agent_trace(seed, drift) for seed in SIM_SEEDS]

    failures = [row for row in failures if len(row.get("messages", [])) > 2]

    pairs = []
    seen: set[str] = set()
    skipped_no_divergence = 0
    skipped_no_replay = 0
    capped_by_template = 0
    per_template: dict[str, int] = {}

    for failure in failures:
        pair = build_pair(failure, drift)
        if pair is None:
            if not (failure.get("tool_calls") or []):
                skipped_no_divergence += 1
            else:
                skipped_no_replay += 1
            continue

        key = _dedupe_key(pair)
        if key in seen:
            continue
        template = pair["provenance"]["template_id"]
        if per_template.get(template, 0) >= MAX_PER_TEMPLATE:
            capped_by_template += 1
            continue

        seen.add(key)
        per_template[template] = per_template.get(template, 0) + 1
        pairs.append(pair)

    # Quotas per label: we only claim to heal a label we have enough rows for.
    counts: dict[str, int] = {}
    for pair in pairs:
        label = pair["provenance"]["label"]
        counts[label] = counts.get(label, 0) + 1

    claimed = {label: n for label, n in counts.items() if n >= MIN_PER_LABEL}
    pairs = [pair for pair in pairs if pair["provenance"]["label"] in claimed]

    # Clean replay mix so the heal does not wipe the clean skill.
    clean_quota = max(
        1, int(len(pairs) * ((CLEAN_MIX_MIN + CLEAN_MIX_MAX) / 2) / (1 - (CLEAN_MIX_MIN + CLEAN_MIX_MAX) / 2))
    )
    clean_traces = []
    for seed in SIM_SEEDS:
        if len(clean_traces) >= clean_quota:
            break
        trace = run_expert(generate_snapshot(seed))
        verified = replay(generate_snapshot(seed), trace["tool_calls"], None)
        if verified["perfect"]:
            trace["provenance"] = {
                "source_run": trace["task_id"],
                "source": "clean_mix",
                "drift": None,
                "seed": seed,
                "replay_hash": hashlib.sha256(
                    json.dumps(trace["tool_calls"], sort_keys=True).encode()
                ).hexdigest()[:16],
                "chosen_reward": verified["reward"],
            }
            clean_traces.append(trace)

    # Poisoned variant: flip the *order* of one chosen/rejected pair in 30% of
    # rows.  A corrupted label, built on purpose — never a corrupted reward.
    poisoned = []
    flip_every = max(1, int(round(1 / 0.30)))
    for index, pair in enumerate(pairs):
        if index % flip_every == 0:
            flipped = json.loads(json.dumps(pair))
            flipped["chosen"], flipped["rejected"] = pair["rejected"], pair["chosen"]
            flipped["provenance"]["poisoned"] = True
            poisoned.append(flipped)
        else:
            poisoned.append(pair)

    chosen_sft = []
    for pair in pairs:
        chosen_sft.append(
            {
                "id": f"sft_{pair['provenance']['seed']}_{pair['provenance']['divergence_step']}",
                "seed": pair["provenance"]["seed"],
                "split": "heal_sft",
                "drift": drift,
                "task": "Find all vulnerable cloud resources.",
                "template_id": TEMPLATE_ID,
                "messages": pair["prompt"] + pair["chosen"],
                "provenance": pair["provenance"],
                "reward": pair["provenance"]["chosen_reward"],
                "success": True,
                "is_correct": True,
            }
        )

    manifest = {
        "drift": drift,
        "source": source,
        "generated_by": "heal.examples",
        "template_id": TEMPLATE_ID,
        "min_per_label": MIN_PER_LABEL,
        "max_per_template": MAX_PER_TEMPLATE,
        "candidate_failures": len(failures),
        "skipped_without_divergence": skipped_no_divergence,
        "skipped_without_replay": skipped_no_replay,
        "capped_by_template": capped_by_template,
        "pairs": len(pairs),
        "clean_mix": len(clean_traces),
        "clean_mix_share": (
            len(clean_traces) / (len(pairs) + len(clean_traces))
            if pairs or clean_traces else 0.0
        ),
        "label_counts": counts,
        "claimed_labels": sorted(claimed),
        "all_chosen_replay_to_1": all(
            pair["provenance"].get("chosen_perfect") for pair in pairs
        ),
        "poisoned_rows": len(poisoned),
    }

    return {
        "pairs": pairs,
        "poisoned": poisoned,
        "chosen_sft": chosen_sft,
        "clean": clean_traces,
        "manifest": manifest,
    }


def write(result: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")

    dump(OUT_DIR / "dpo_pairs.jsonl", result["pairs"])
    dump(OUT_DIR / "dpo_pairs_poisoned.jsonl", result["poisoned"])
    dump(OUT_DIR / "sft_chosen.jsonl", result["chosen_sft"])
    dump(OUT_DIR / "clean_mix.jsonl", result["clean"])

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(result["manifest"], indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", default="D1", choices=["D1", "D2", "D3", "D4"])
    parser.add_argument(
        "--source", default="simulator", choices=["simulator", "rollouts"]
    )
    args = parser.parse_args()

    result = build(args.drift, args.source)
    manifest = result["manifest"]

    if manifest["pairs"] == 0:
        print(
            "No minimal-edit pair could be built. "
            f"{manifest['skipped_without_divergence']} failures had no tool call to "
            "diverge from. That is the honest answer: minimal-edit DPO needs an "
            "agent that already calls tools."
        )
        return

    if not manifest["all_chosen_replay_to_1"]:
        raise SystemExit("a chosen trace did not replay to reward 1 — refusing to emit")

    write(result)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
