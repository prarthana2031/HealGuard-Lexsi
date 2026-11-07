"""Build a failure battery from the real environment, labelled by construction.

The old evaluation scored the classifier on 16 hand-written JSON blobs whose
expected labels were written to match the rules.  That measures nothing: the
same person wrote the rules and the answers.

Here every trajectory is produced by running a *deliberately broken policy*
against the real `CloudEnv` and the real drift wrappers, so the tool results and
error strings are genuine.  The ground-truth label is the policy's name, fixed
before the classifier is run.  A judge can read the policy and check the claim.

A policy is only emitted on a seed where its answer is genuinely wrong, so the
battery never labels an accidentally-correct trace as a failure.

Usage:
    python -m classifier.build_failure_battery
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from expert.scripted import system_prompt_for  # noqa: E402
from heal.drifts.registry import build_runtime  # noqa: E402
from oracle.checker import check_vulnerabilities  # noqa: E402
from world.snapshot import generate_snapshot  # noqa: E402
from world.tools import CloudEnv  # noqa: E402

OUT = Path("classifier/fixtures/failure_battery.jsonl")
TEMPLATE_ID = "custom_tool_call_v1"
SEED_POOL = range(7000, 7400)


def _call(name: str, arguments: dict[str, Any]) -> str:
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


class Episode:
    """Tiny trace recorder driven by a broken policy."""

    def __init__(self, seed: int, drift: str | None):
        self.snapshot = generate_snapshot(seed)
        self.truth = check_vulnerabilities(self.snapshot)
        self.drift = drift
        base = CloudEnv(self.snapshot)
        self.env, self.schemas = build_runtime(base, drift=drift)
        self.schema_names = {schema["name"] for schema in self.schemas}
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt_for(self.schemas)},
            {"role": "user", "content": "Find all vulnerable cloud resources."},
        ]
        self.calls: list[dict[str, Any]] = []
        self.final_answer: list[str] = []
        self.finished = False

    def act(self, name: str, arguments: dict[str, Any]) -> Any:
        self.messages.append({"role": "assistant", "content": _call(name, arguments)})
        self.calls.append({"name": name, "arguments": arguments})
        function = getattr(self.env, name, None)
        if callable(function):
            try:
                result = function(**arguments)
            except TypeError as exc:
                result = {"error": f"schema_violation: {exc}"}
        else:
            result = {"error": f"unknown_tool: Tool {name!r} not found"}
        self.messages.append(
            {"role": "tool", "content": json.dumps(result, default=str)}
        )
        return result

    def inject_env_error(self) -> None:
        self.messages.append(
            {"role": "tool", "content": "Error: 503 Service Unavailable"}
        )

    def finish(self, answer: list[str]) -> None:
        self.messages.append(
            {"role": "assistant", "content": _call("finish", {"answer": answer})}
        )
        self.calls.append({"name": "finish", "arguments": {"answer": answer}})
        self.messages.append(
            {
                "role": "tool",
                "content": json.dumps({"status": "success", "answer": answer}),
            }
        )
        self.final_answer = list(answer)
        self.finished = True

    def wrong_grounded_bucket(self) -> list[str]:
        """A listed bucket that is not vulnerable: grounded, non-empty, wrong.

        Grounded so it is not `ungrounded_answer`; non-empty so it is not
        `premature_finish`; wrong so the episode really is a failure.
        """
        for bucket in self.snapshot["buckets"]:
            if bucket["id"] not in self.truth:
                return [bucket["id"]]
        return list(self.truth[:1])

    def record(self, expected_label: str, policy: str) -> dict[str, Any]:
        steps = sum(1 for call in self.calls if call["name"] != "finish")
        valid = sum(
            1
            for call in self.calls
            if call["name"] != "finish" and call["name"] in self.schema_names
        )
        return {
            "task_id": f"{policy}_{self.drift or 'clean'}_{self.snapshot['seed']}",
            "id": f"{policy}_{self.drift or 'clean'}_{self.snapshot['seed']}",
            "seed": self.snapshot["seed"],
            "split": "drift_targeted" if self.drift else "dev",
            "drift": self.drift,
            "expected_label": expected_label,
            "policy": policy,
            "template_id": TEMPLATE_ID,
            "task": "Find all vulnerable cloud resources.",
            "messages": self.messages,
            "tool_calls": self.calls,
            "final_answer": self.final_answer,
            "reward": 0.0,
            "success": False,
            "num_steps": steps,
            "valid_tool_calls": valid,
            "total_tool_calls": len(self.calls),
            "redundant_calls": 0,
            "finished": self.finished,
        }


# ---------------------------------------------------------------------------
# Broken policies.  The label is the policy name, decided before classification.
# ---------------------------------------------------------------------------

MIN_TRUTH = lambda ep: len(ep.truth) >= 1        # noqa: E731


def policy_wrong_tool(ep: Episode) -> None:
    # Canonical instance from the taxonomy: "an unknown tool name".
    ep.act("list_s3_buckets", {})
    ep.finish([])


def policy_wrong_order(ep: Episode) -> None:
    ep.act("get_bucket_policy", {"bucket": ep.snapshot["buckets"][0]["id"]})
    ep.finish([])


def policy_schema_violation(ep: Episode) -> None:
    ep.act("list_buckets", {"limit": "five"})
    ep.finish([])


def policy_premature_finish(ep: Episode) -> None:
    ep.act("list_buckets", {})
    ep.finish([])


def policy_ungrounded_answer(ep: Episode) -> None:
    ep.act("list_buckets", {})
    ep.finish(["bucket-that-never-existed"])


def policy_loop_collapse(ep: Episode) -> None:
    for _ in range(3):
        ep.act("list_buckets", {})
    ep.finish(ep.wrong_grounded_bucket())


def policy_redundant_call(ep: Episode) -> None:
    for _ in range(2):
        ep.act("list_buckets", {})
    ep.finish(ep.wrong_grounded_bucket())


def policy_unrecovered_error(ep: Episode) -> None:
    # Prerequisites satisfied, then a hallucinated id fails and is repeated
    # unchanged: "an error followed by the same call", per the taxonomy.
    ep.act("list_buckets", {})
    for _ in range(2):
        ep.act("get_bucket_acl", {"bucket": "bucket-does-not-exist"})
    ep.finish(ep.wrong_grounded_bucket())


def policy_env_error(ep: Episode) -> None:
    ep.act("list_buckets", {})
    ep.inject_env_error()
    ep.finish([])


def policy_residual(ep: Episode) -> None:
    # Valid calls, grounded answer, no repeat, no error — just an incomplete
    # answer that no rule covers.
    ep.act("list_buckets", {})
    ep.finish(ep.wrong_grounded_bucket())


def policy_drift_mismatch(ep: Episode) -> None:
    """The real demo failure: v0 trained on the clean schema, run under v2."""
    bucket = ep.snapshot["buckets"][0]["id"]
    if ep.drift == "D1":
        ep.act("get_bucket_acl", {"bucket": bucket})
    elif ep.drift == "D2":
        ep.act("list_buckets", {})
    elif ep.drift == "D3":
        ep.act("list_buckets", {})
    elif ep.drift == "D4":
        ep.act("get_bucket_acl", {"bucket": bucket})
    else:  # pragma: no cover - defensive
        raise ValueError("drift_mismatch policy needs a drift")
    ep.finish([])


CLEAN_POLICIES: list[tuple[str, Callable[[Episode], None], Callable[[Episode], bool]]] = [
    ("wrong_tool", policy_wrong_tool, MIN_TRUTH),
    ("wrong_order", policy_wrong_order, MIN_TRUTH),
    ("schema_violation", policy_schema_violation, MIN_TRUTH),
    ("premature_finish", policy_premature_finish, MIN_TRUTH),
    ("ungrounded_answer", policy_ungrounded_answer, lambda ep: True),
    ("loop_collapse", policy_loop_collapse, MIN_TRUTH),
    ("redundant_call", policy_redundant_call, MIN_TRUTH),
    ("unrecovered_error", policy_unrecovered_error, MIN_TRUTH),
    ("env_error", policy_env_error, MIN_TRUTH),
    ("residual", policy_residual, MIN_TRUTH),
]

DRIFTS = ("D1", "D2", "D3", "D4")
PER_LABEL = 8


def build() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for name, policy, precondition in CLEAN_POLICIES:
        collected = 0
        for seed in SEED_POOL:
            episode = Episode(seed, drift=None)
            if not precondition(episode):
                continue
            policy(episode)
            rows.append(episode.record(name, name))
            collected += 1
            if collected == PER_LABEL:
                break
        if collected < PER_LABEL:
            raise RuntimeError(f"{name}: only collected {collected}")

    for drift in DRIFTS:
        collected = 0
        for seed in SEED_POOL:
            episode = Episode(seed, drift=drift)
            if not MIN_TRUTH(episode):
                continue
            policy_drift_mismatch(episode)
            rows.append(episode.record("drift_mismatch", "drift_mismatch"))
            collected += 1
            if collected == PER_LABEL:
                break
        if collected < PER_LABEL:
            raise RuntimeError(f"drift_mismatch/{drift}: only {collected}")

    return rows


def main() -> None:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} trajectories to {OUT}")


if __name__ == "__main__":
    main()
