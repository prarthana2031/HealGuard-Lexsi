"""What counts as having actually done the audit.

The oracle scores the *answer*, which is right: a task's success is a fact about
the cloud account, not about effort. But "successful" is not the same as
"grounded", and conflating them produced a real error in this project.

Measured on the v0 rollouts under D1 (seeds 30000-30059): of 37 successes, only
**5 had made every required evidence call**. The other 32 answered correctly
without reading all the resources — and took a median of 12 steps against 14 for
the grounded ones. So a gate that compares step counts over *all* successes is
comparing luck against work, and it penalised the healed model for doing the
reads the broken baseline skipped.

This module is the definition both the rollout harness and the gate use, so
"grounded" cannot mean two different things in two places.
"""

from __future__ import annotations

from typing import Any

# A renamed tool still counts as the tool it replaced.
TOOL_ALIASES = {"get_acl": "get_bucket_acl"}

BUCKET_TOOLS = (
    "get_bucket_acl",
    "get_bucket_policy",
    "get_public_access_block",
)

ENUMERATION_CALLS = ("list_buckets", "list_iam_roles", "describe_security_groups")


def normalize_tool(name: str | None) -> str:
    return TOOL_ALIASES.get(name or "", name or "")


def target_of(arguments: Any) -> str:
    """The resource a call is about, whichever spelling the API uses."""
    if not isinstance(arguments, dict):
        return ""
    return (
        arguments.get("bucket_name")
        or arguments.get("bucket")
        or arguments.get("role_name")
        or ""
    )


def required_evidence(snapshot: dict[str, Any]) -> set[tuple[str, str]]:
    """Every call a correct audit has to make to be able to justify its answer."""
    needed: set[tuple[str, str]] = {(call, "") for call in ENUMERATION_CALLS}
    for bucket in snapshot["buckets"]:
        for tool in BUCKET_TOOLS:
            needed.add((tool, bucket["id"]))
    for role in snapshot["roles"]:
        needed.add(("get_role_policies", role["id"]))
    return needed


def observed_evidence(calls: list[dict[str, Any]]) -> set[tuple[str, str]]:
    observed = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("name") == "finish":
            continue
        observed.add((normalize_tool(call.get("name")), target_of(call.get("arguments"))))
    return observed


def missing_evidence(
    snapshot: dict[str, Any], calls: list[dict[str, Any]]
) -> set[tuple[str, str]]:
    return required_evidence(snapshot) - observed_evidence(calls)


def is_grounded(
    snapshot: dict[str, Any],
    calls: list[dict[str, Any]],
    success: bool,
) -> bool:
    """Correct *and* every required evidence call made."""
    return bool(success) and not missing_evidence(snapshot, calls)
