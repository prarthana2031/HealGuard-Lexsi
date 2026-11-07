"""Run one episode under a named policy and emit a gate-ready record.

Three policies, all deterministic and all executable without a GPU:

`stale`     what a LoRA trained on the clean API does when the API moves: it
            still speaks the old argument names, still reads `items`, still
            omits `region`.  This is the baseline side of the gate.
`expert`    the scripted expert under the current schema.  This is the ceiling
            and the candidate band for pipeline validation.
`poisoned`  the expert with the answer forced to the empty set — what a
            corrupted retrain looks like.  The gate must reject it.

A policy run is *not* a model result.  It validates that the pipeline, the
oracle, the classifier and the gate are wired correctly before the GPU numbers
arrive.  Records always say which policy produced them.
"""

from __future__ import annotations

import json
from typing import Any

from expert.scripted import family_success, run_expert, system_prompt_for
from heal.drifts.registry import build_runtime
from heal.replay import replay
from oracle.checker import check_vulnerabilities
from world.snapshot import generate_snapshot
from world.tools import CloudEnv

POLICIES = ("stale", "expert", "poisoned")


def _call(name: str, arguments: dict[str, Any]) -> str:
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


def run_stale(seed: int, drift: str | None) -> dict[str, Any]:
    """v0: competent on the clean API, stale against v2.

    It reasons correctly from the observations it manages to collect, so on the
    clean API it scores ~1.0.  Under a drift it degrades for a *specific*
    reason: the bucket tools reject the old argument name, `list_buckets` no
    longer returns `items`, `region` is missing, or a tool was renamed.  That is
    what makes "each drift costs at least 20 points" measurable instead of
    assumed.
    """
    snapshot = generate_snapshot(seed)
    truth = check_vulnerabilities(snapshot)
    env, schemas = build_runtime(CloudEnv(snapshot), drift=drift)

    messages = [
        {"role": "system", "content": system_prompt_for(schemas)},
        {"role": "user", "content": "Find all vulnerable cloud resources."},
    ]
    calls: list[dict[str, Any]] = []
    vulnerable: list[str] = []

    def act(name: str, arguments: dict[str, Any]) -> Any:
        messages.append({"role": "assistant", "content": _call(name, arguments)})
        calls.append({"name": name, "arguments": arguments})
        function = getattr(env, name, None)
        if callable(function):
            try:
                result = function(**arguments)
            except TypeError as exc:
                result = {"error": f"schema_violation: {exc}"}
        else:
            result = {"error": f"unknown_tool: Tool {name!r} not found"}
        messages.append({"role": "tool", "content": json.dumps(result, default=str)})
        return result

    def failed(result: Any) -> bool:
        return not isinstance(result, dict) or "error" in result

    # --- F1: buckets ------------------------------------------------------
    listing = act("list_buckets", {})
    # v0 only knows the clean envelope key; D2's `bucket_list` is invisible.
    buckets = (
        listing.get("items", [])
        if isinstance(listing, dict) and not failed(listing)
        else []
    )
    for bucket in buckets:
        acl = act("get_bucket_acl", {"bucket": bucket})
        policy = act("get_bucket_policy", {"bucket": bucket})
        block = act("get_public_access_block", {"bucket": bucket})

        # v0 cannot evaluate a bucket it could not read.
        if failed(acl) or failed(policy) or failed(block):
            continue
        if block.get("public_access_block"):
            continue
        if policy.get("policy_deny"):
            continue
        if acl.get("acl_public") or policy.get("policy_public"):
            vulnerable.append(bucket)

    # --- F2: roles --------------------------------------------------------
    roles = act("list_iam_roles", {})
    for role in (roles.get("roles", []) if not failed(roles) else []):
        policies = act("get_role_policies", {"role_name": role})
        if failed(policies):
            continue
        for policy in policies.get("policies", []):
            if (
                policy.get("effect") == "Allow"
                and policy.get("action") == "*"
                and policy.get("resource") == "*"
            ):
                vulnerable.append(role)
                break

    # --- F4: security groups ---------------------------------------------
    groups = act("describe_security_groups", {})
    for group in (
        groups.get("security_groups", []) if not failed(groups) else []
    ):
        for rule in group.get("ingress", []):
            if rule.get("cidr") == "0.0.0.0/0" and rule.get("port") in (22, 3389, 3306):
                vulnerable.append(group["id"])
                break

    answer = sorted(set(vulnerable))
    messages.append(
        {"role": "assistant", "content": _call("finish", {"answer": answer})}
    )
    calls.append({"name": "finish", "arguments": {"answer": answer}})

    return {
        "id": f"{drift or 'clean'}_{seed}",
        "task_id": f"{drift or 'clean'}_{seed}",
        "seed": seed,
        "drift": drift,
        "policy": "stale",
        "messages": messages,
        "tool_calls": calls,
        "final_answer": answer,
        "family_success": family_success(answer, truth),
        "template_id": "custom_tool_call_v1",
    }


def run_poisoned(seed: int, drift: str | None) -> dict[str, Any]:
    """The expert with a poisoned finish step: a corrupted retrain."""
    trace = run_expert(generate_snapshot(seed), drift=drift)
    trace["tool_calls"] = [
        call
        for call in trace["tool_calls"]
        if call["name"] != "finish"
    ] + [{"name": "finish", "arguments": {"answer": []}}]
    for message in reversed(trace["messages"]):
        if message.get("role") == "assistant" and "finish" in message.get("content", ""):
            message["content"] = _call("finish", {"answer": []})
            break
    trace["final_answer"] = []
    trace["answer"] = []
    trace["policy"] = "poisoned"
    trace["id"] = f"{drift or 'clean'}_{seed}"
    trace["task_id"] = trace["id"]
    trace["family_success"] = family_success(
        [], check_vulnerabilities(generate_snapshot(seed))
    )
    return trace


def run_episode(seed: int, drift: str | None, policy: str, bucket: str) -> dict[str, Any]:
    if policy == "expert":
        trace = run_expert(generate_snapshot(seed), drift=drift)
        trace["policy"] = "expert"
    elif policy == "poisoned":
        trace = run_poisoned(seed, drift)
    elif policy == "stale":
        trace = run_stale(seed, drift)
    else:
        raise ValueError(policy)

    from classifier.rules import classify

    verified = replay(generate_snapshot(seed), trace["tool_calls"], drift)
    classification = classify({**trace, "success": verified["correct"]})

    redundant = 0
    seen = set()
    steps = 0
    for call in trace["tool_calls"]:
        if call["name"] == "finish":
            continue
        steps += 1
        signature = json.dumps(call, sort_keys=True)
        if signature in seen:
            redundant += 1
        seen.add(signature)

    task_id = f"{drift or 'clean'}_{seed}"

    return {
        "id": task_id,
        "task_id": task_id,
        "seed": seed,
        "is_none_task": not check_vulnerabilities(generate_snapshot(seed)),
        "bucket": bucket,
        "drift": drift or "clean",
        "policy": policy,
        "success": verified["correct"],
        "num_steps": steps,
        "total_tool_calls": len(trace["tool_calls"]),
        "redundant_calls": redundant,
        "family_success": trace["family_success"],
        "failure_mode": classification["primary"],
        "reward": verified["reward"],
        "drift_inferred": classification["drift_inferred"],
    }


def failure_traces(bucket: str, policy: str = "stale") -> list[dict]:
    """Full traces (with messages) for the failures in a bucket."""
    traces = []
    for seed, drift in bucket_seeds():
        trace = run_stale(seed, drift) if policy == "stale" else run_expert(
            generate_snapshot(seed), drift=drift
        )
        verified = replay(generate_snapshot(seed), trace["tool_calls"], drift)
        if verified["correct"]:
            continue
        trace["success"] = False
        trace["drift"] = drift
        traces.append(trace)
    return traces


def bucket_seeds() -> list[tuple[int, str | None]]:
    import json as _json
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    entries: list[tuple[int, str | None]] = []
    for relative in (
        "data/drift_targeted/expert_traces.jsonl",
        "data/gate_regression/expert_traces.jsonl",
    ):
        path = root / relative
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    row = _json.loads(line)
                    entries.append((row["seed"], row.get("drift")))
    return entries


def write_records(
    seeds: list[tuple[int, str | None]],
    policy: str,
    bucket: str,
    path: str,
) -> list[dict[str, Any]]:
    records = [run_episode(seed, drift, policy, bucket) for seed, drift in seeds]
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    return records
