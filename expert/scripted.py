"""Deterministic scripted expert.

The expert is the teacher for v0 SFT, so the trace has to be exactly the format
the trainer and the scorers consume.  Two things were wrong before and are
fixed here:

1. `inject_error` used to append the *same* tool call twice after a 503.  That
   taught the student that the correct response to an error is to repeat the
   identical call — the exact `unrecovered_error` / `redundant_call` behaviour
   the project is supposed to fix.  The retry now changes the call.
2. The trace carried only `task_id`, `final_answer` and `is_correct`, so the
   scorer could not read it without an adapter.  It now carries the full
   `agent/trajectory_schema.md` contract, including per-family success, which is
   what makes a per-family regression table possible.
"""

from __future__ import annotations

import json
from typing import Any

from heal.drifts.dispatcher import create_dispatcher
from oracle.checker import check_vulnerabilities
from world.tools import TOOL_SCHEMAS, CloudEnv

TEMPLATE_ID = "custom_tool_call_v1"
FAMILIES = ("F1", "F2", "F4")


def system_prompt_for(schemas: list[dict[str, str]]) -> str:
    """The exact system prompt the expert and v0 see."""
    tool_desc = "\n".join(
        f"- {tool['name']}: {tool['description']}" for tool in schemas
    )
    return (
        "You are a cloud auditor. Output exactly like this: <tool_call>\n"
        '{"name": "tool_name", "arguments": {...}}\n'
        "</tool_call>\nAvailable Tools:\n"
        f"{tool_desc}"
    )


def render_call(name: str, arguments: dict[str, Any]) -> str:
    """The canonical assistant turn for one tool call.

    One renderer for the expert, the rollout harness and the example generator.
    DPO pairs are built from *calls*, not from raw model text, so this has to be
    the only spelling of a call in the codebase.
    """
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


def _call(name: str, arguments: dict[str, Any]) -> str:
    return render_call(name, arguments)


def family_of(resource_id: str) -> str:
    if resource_id.startswith("bucket-"):
        return "F1"
    if resource_id.startswith("role-"):
        return "F2"
    return "F4"


def family_success(answer: list[str], truth: list[str]) -> dict[str, bool]:
    """Per-family exact-match, so a bad average cannot hide a broken family."""
    answer_by_family: dict[str, set[str]] = {family: set() for family in FAMILIES}
    truth_by_family: dict[str, set[str]] = {family: set() for family in FAMILIES}

    for resource in answer:
        answer_by_family[family_of(resource)].add(resource)
    for resource in truth:
        truth_by_family[family_of(resource)].add(resource)

    return {
        family: answer_by_family[family] == truth_by_family[family]
        for family in FAMILIES
    }


class TraceBuilder:
    """Accumulates one expert episode with the full metric contract."""

    def __init__(self, snapshot: dict[str, Any], drift: str | None):
        self.snapshot = snapshot
        self.drift = drift
        self.truth = check_vulnerabilities(snapshot)
        base_env = CloudEnv(snapshot)
        dispatcher = create_dispatcher(base_env, drift=drift)
        self.env = dispatcher.env
        self.schemas = dispatcher.schemas
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt_for(self.schemas)},
            {"role": "user", "content": "Find all vulnerable cloud resources."},
        ]
        self.calls: list[dict[str, Any]] = []
        self.error_results: list[int] = []

    # -- actions ---------------------------------------------------------
    def act(self, name: str, arguments: dict[str, Any]) -> Any:
        self.messages.append({"role": "assistant", "content": _call(name, arguments)})
        self.calls.append({"name": name, "arguments": arguments})

        function = getattr(self.env, name, None)
        if not callable(function):
            result: Any = {"error": f"unknown_tool: Tool {name!r} not found"}
        else:
            try:
                result = function(**arguments)
            except TypeError as exc:
                result = {"error": f"schema_violation: {exc}"}

        if isinstance(result, dict) and "error" in result:
            self.error_results.append(len(self.calls) - 1)

        self.messages.append(
            {"role": "tool", "content": json.dumps(result, default=str)}
        )
        return result

    def finish(self) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": _call("finish", {"answer": self.truth}),
            }
        )
        self.calls.append({"name": "finish", "arguments": {"answer": self.truth}})
        self.messages.append(
            {
                "role": "tool",
                "content": json.dumps({"status": "success", "answer": self.truth}),
            }
        )
        self.env.finish(self.truth)

    # -- record ----------------------------------------------------------
    def record(self, inject_error: bool) -> dict[str, Any]:
        redundant = 0
        seen: set[str] = set()
        for call in self.calls:
            if call["name"] == "finish":
                continue
            signature = json.dumps(call, sort_keys=True)
            if signature in seen:
                redundant += 1
            seen.add(signature)

        steps = sum(1 for call in self.calls if call["name"] != "finish")
        return {
            "id": f"task_{self.snapshot['seed']}",
            "task_id": f"task_{self.snapshot['seed']}",
            "seed": self.snapshot["seed"],
            "split": "train_clean" if self.drift is None else "drift_targeted",
            "drift": self.drift,
            "task": "Find all vulnerable cloud resources.",
            "template_id": TEMPLATE_ID,
            "messages": self.messages,
            "tool_calls": self.calls,
            "required_tools": [
                schema["name"] for schema in self.schemas if schema["name"] != "finish"
            ],
            "final_answer": self.truth,
            "answer": self.truth,
            "family_success": family_success(self.truth, self.truth),
            "reward": 1.0,
            "success": True,
            "is_correct": True,
            "num_steps": steps,
            "valid_tool_calls": len(self.calls) - len(self.error_results),
            "total_tool_calls": len(self.calls),
            "redundant_calls": redundant,
            "finished": True,
            "error": None,
            "injected_error": inject_error,
        }


def run_expert(
    snapshot: dict[str, Any],
    inject_error: bool = False,
    drift: str | None = None,
) -> dict[str, Any]:
    """Run the expert once. Deterministic: same snapshot in, same trace out."""

    trace = TraceBuilder(snapshot, drift)

    # Argument names move with the API version, exactly as a trained model's
    # would have to.  The expert is allowed to know the new schema.
    bucket_arg = "bucket_name" if drift == "D1" else "bucket"
    region = {"region": "us-east-1"} if drift == "D3" else {}
    acl_tool = "get_acl" if drift == "D4" else "get_bucket_acl"

    def bucket_args(bucket_id: str) -> dict[str, Any]:
        return {bucket_arg: bucket_id}

    # 1. List buckets.
    if inject_error:
        # The first attempt really fails with a 503.  The retry that follows
        # changes the call (`limit` 5) so the student learns recovery instead
        # of "repeat the identical call", which is the failure we are fixing.
        trace.messages.append(
            {"role": "assistant", "content": _call("list_buckets", dict(region))}
        )
        trace.calls.append({"name": "list_buckets", "arguments": dict(region)})
        trace.error_results.append(len(trace.calls) - 1)
        trace.messages.append(
            {"role": "tool", "content": "Error: 503 Service Unavailable"}
        )
        listing = trace.act("list_buckets", {"limit": 5, **region})
    else:
        listing = trace.act("list_buckets", dict(region))

    # 2. Pagination, then every bucket.
    buckets = listing.get("items", listing.get("bucket_list", [])) if isinstance(listing, dict) else list(listing)
    while isinstance(listing, dict) and (
        listing.get("next_token") or listing.get("continuation")
    ):
        token = listing.get("next_token") or listing.get("continuation")
        listing = trace.act("list_buckets", {"next_token": token, **region})
        buckets.extend(
            listing.get("items", listing.get("bucket_list", []))
            if isinstance(listing, dict)
            else list(listing)
        )

    for bucket_id in buckets:
        trace.act(acl_tool, bucket_args(bucket_id))
        trace.act("get_bucket_policy", bucket_args(bucket_id))
        trace.act("get_public_access_block", bucket_args(bucket_id))

    # 3. IAM roles.
    roles = trace.act("list_iam_roles", {})
    for role_id in roles.get("roles", []):
        trace.act("get_role_policies", {"role_name": role_id})

    # 4. Security groups.
    trace.act("describe_security_groups", dict(region))

    # 5. Finish.
    trace.finish()

    return trace.record(inject_error=inject_error)


if __name__ == "__main__":
    from world.snapshot import generate_snapshot

    print(
        json.dumps(
            run_expert(generate_snapshot(0), inject_error=True)["num_steps"]
        )
    )
