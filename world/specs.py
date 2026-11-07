"""Strict tool-argument specifications shared by the world, classifier and healer.

One table, one source of truth.  The environment rejects a call that does not
match it; the classifier names the violation; the healer builds the corrected
call from it.
"""

from __future__ import annotations

import inspect
from typing import Any

from world.tools import CloudEnv

# Argument name -> accepted python types.  `None` in the value tuple means the
# argument is optional.
TOOL_ARG_SPECS: dict[str, dict[str, tuple[type, ...]]] = {
    "list_buckets": {
        "limit": (int,),
        "next_token": (str, type(None)),
    },
    "get_bucket_acl": {"bucket": (str,)},
    "get_bucket_policy": {"bucket": (str,)},
    "get_public_access_block": {"bucket": (str,)},
    "list_iam_roles": {"limit": (int,)},
    "get_role_policies": {"role_name": (str,)},
    "describe_security_groups": {"group_ids": (list, type(None))},
    "finish": {"answer": (list,)},
}

# Arguments that are required.  Everything not listed here is optional.
REQUIRED_ARGS: dict[str, frozenset[str]] = {
    "list_buckets": frozenset(),
    "get_bucket_acl": frozenset({"bucket"}),
    "get_bucket_policy": frozenset({"bucket"}),
    "get_public_access_block": frozenset({"bucket"}),
    "list_iam_roles": frozenset(),
    "get_role_policies": frozenset({"role_name"}),
    "describe_security_groups": frozenset(),
    "finish": frozenset({"answer"}),
}

BUCKET_TOOLS = (
    "get_bucket_acl",
    "get_bucket_policy",
    "get_public_access_block",
)

# Tools whose results must exist before a dependent call is legal.
PREREQUISITES: dict[str, str] = {
    "get_bucket_acl": "list_buckets",
    "get_bucket_policy": "list_buckets",
    "get_public_access_block": "list_buckets",
    "get_role_policies": "list_iam_roles",
}


def arg_names(tool: str) -> set[str]:
    return set(TOOL_ARG_SPECS.get(tool, {}))


def is_valid_type(value: Any, allowed: tuple[type, ...]) -> bool:
    if isinstance(value, bool) and bool not in allowed:
        return False
    return isinstance(value, allowed)


def check_arguments(tool: str, arguments: Any) -> list[str]:
    """Return a list of schema violations, empty when the call is well formed."""
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]

    spec = TOOL_ARG_SPECS.get(tool)
    if spec is None:
        return []  # unknown tool is handled by the wrong_tool rule

    problems: list[str] = []

    missing = REQUIRED_ARGS.get(tool, frozenset()) - arguments.keys()
    for name in sorted(missing):
        problems.append(f"missing required argument {name!r}")

    for name, value in arguments.items():
        if name not in spec:
            problems.append(f"unexpected argument {name!r}")
            continue
        if value is None and type(None) in spec[name]:
            continue
        if not is_valid_type(value, spec[name]):
            expected = "/".join(t.__name__ for t in spec[name])
            problems.append(
                f"argument {name!r} must be {expected}, got "
                f"{type(value).__name__}"
            )

    return problems


def env_signatures() -> dict[str, inspect.Signature]:
    """Live signatures, used by a test that keeps this table honest."""
    return {
        name: inspect.signature(getattr(CloudEnv, name))
        for name in TOOL_ARG_SPECS
        if hasattr(CloudEnv, name)
    }
