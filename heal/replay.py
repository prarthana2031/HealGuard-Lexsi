"""Replay a trace in the real environment and score it with the plan's reward.

The plan's non-negotiable core is "corrective examples that all replay to reward
1".  That is only meaningful if replay actually re-executes every tool call
against the drifted environment and recomputes the oracle, so this module does
exactly that — it never trusts the `reward` field already in a file.

Reward, from section 9 of the plan:

    R = 1[oracle correct] * (1 - lambda * min(1, steps / step_budget)) + shaping
    shaping = 0.05 * valid_call_rate
              - 0.05 * 1[repeated identical call]
              - 0.10 * 1[loop]

Shaping is paid only on a correct answer and its absolute value is capped at
0.15, so format credit can never outrank a wrong answer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from heal.drifts.registry import build_runtime
from oracle.checker import check_vulnerabilities
from world.tools import CloudEnv

LAMBDA = 0.15
STEP_BUDGET = 24
SHAPING_CAP = 0.15

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for block in _CALL_RE.findall(message.get("content", "") or ""):
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                calls.append({"name": None, "arguments": None, "malformed": block})
                continue
            calls.append(
                {
                    "name": parsed.get("name"),
                    "arguments": parsed.get("arguments", {}),
                }
            )
    return calls


def _loop_detected(signatures: list[str]) -> bool:
    counts: dict[str, int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
        if counts[signature] >= 3:
            return True
    return False


def replay(
    snapshot: dict[str, Any],
    calls: list[dict[str, Any]],
    drift: str | None,
) -> dict[str, Any]:
    """Execute the calls against a fresh environment and judge the answer."""

    truth = check_vulnerabilities(snapshot)
    env, _schemas = build_runtime(CloudEnv(snapshot), drift=drift)

    answer: list[str] | None = None
    valid = 0
    attempted = 0
    signatures: list[str] = []
    repeated = False
    errors: list[str] = []

    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            errors.append("malformed call")
            continue

        if name == "finish":
            value = arguments.get("answer")
            answer = [str(item) for item in value] if isinstance(value, list) else []
            continue

        attempted += 1
        signature = json.dumps({"name": name, "arguments": arguments}, sort_keys=True)
        if signature in signatures:
            repeated = True
        signatures.append(signature)

        function = getattr(env, name, None)
        if not callable(function):
            errors.append(f"unknown_tool:{name}")
            continue
        try:
            result = function(**arguments)
        except TypeError as exc:
            errors.append(f"schema_violation:{exc}")
            continue

        if isinstance(result, dict) and "error" in result:
            errors.append(str(result["error"]))
        else:
            valid += 1

    if answer is None:
        answer = []

    correct = sorted(answer) == sorted(truth)
    steps = attempted
    valid_rate = valid / attempted if attempted else 0.0

    shaping = 0.0
    if correct:
        shaping = 0.05 * valid_rate
        if repeated:
            shaping -= 0.05
        if _loop_detected(signatures):
            shaping -= 0.10
        shaping = max(-SHAPING_CAP, min(SHAPING_CAP, shaping))

    reward = (
        (1.0 - LAMBDA * min(1.0, steps / STEP_BUDGET)) if correct else 0.0
    ) + shaping

    loop = _loop_detected(signatures)

    # "Replays to reward 1" means the *task* reward is perfect: the oracle set
    # matches and nothing was wasted.  The scalar R also carries the step
    # penalty, so a multi-step success can never equal 1.0 exactly.
    perfect = correct and not errors and not repeated and not loop

    return {
        "correct": correct,
        "perfect": perfect,
        "answer": answer,
        "truth": truth,
        "reward": round(reward, 6),
        "steps": steps,
        "valid_calls": valid,
        "attempted_calls": attempted,
        "valid_call_rate": round(valid_rate, 6),
        "repeated_call": repeated,
        "loop": loop,
        "errors": errors,
    }


def replay_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Replay a stored trajectory in the environment it claims to have run in."""
    from world.snapshot import generate_snapshot

    snapshot = trajectory.get("snapshot") or generate_snapshot(trajectory["seed"])
    calls = trajectory.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        calls = parse_calls(trajectory.get("messages", []))
    return replay(snapshot, calls, trajectory.get("drift"))
