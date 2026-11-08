"""Machine-readable schema diffs for D1-D4.

The plan says `drift_mismatch` is "a schema or unknown-tool error whose name or
argument is in the schema diff".  That only works if the schema diff is a real
object the classifier can read, not a comment in a wrapper.  This module is the
single source of truth for the four injected drifts.

Each entry describes what the API v2 changed relative to the clean API:

  renamed_args      old argument name -> new argument name, per tool
  renamed_tools     old tool name -> new tool name
  required_args     arguments that are newly required, per tool
  renamed_fields    old response field -> new response field, per tool

`classify` uses these to decide between `drift_mismatch`, `schema_violation`,
`wrong_tool` and `ungrounded_answer`.  `heal.examples` uses the same object to
build the corrective example, so the label and the fix cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SchemaDiff:
    drift_id: str
    description: str
    renamed_args: dict[str, dict[str, str]] = field(default_factory=dict)
    renamed_tools: dict[str, str] = field(default_factory=dict)
    required_args: dict[str, tuple[str, ...]] = field(default_factory=dict)
    renamed_fields: dict[str, dict[str, str]] = field(default_factory=dict)

    # -- helpers -----------------------------------------------------------

    def old_arg_names(self) -> set[str]:
        return {
            old
            for per_tool in self.renamed_args.values()
            for old in per_tool
        }

    def new_arg_names(self) -> set[str]:
        return {
            new
            for per_tool in self.renamed_args.values()
            for new in per_tool.values()
        }

    def arg_rename_for(self, tool: str) -> dict[str, str]:
        return dict(self.renamed_args.get(tool, {}))

    def names(self) -> set[str]:
        """Every argument/tool name involved in this diff, old or new."""
        return (
            self.old_arg_names()
            | self.new_arg_names()
            | set(self.renamed_tools)
            | set(self.renamed_tools.values())
            | {
                arg
                for args in self.required_args.values()
                for arg in args
            }
            | {
                name
                for per_tool in self.renamed_fields.values()
                for pair in per_tool.items()
                for name in pair
            }
        )


SCHEMA_DIFFS: dict[str, SchemaDiff] = {
    "D1": SchemaDiff(
        drift_id="D1",
        description=(
            "The bucket argument is renamed to bucket_name on the three "
            "bucket tools. No world fact changes."
        ),
        renamed_args={
            "get_bucket_acl": {"bucket": "bucket_name"},
            "get_bucket_policy": {"bucket": "bucket_name"},
            "get_public_access_block": {"bucket": "bucket_name"},
        },
    ),
    "D2": SchemaDiff(
        drift_id="D2",
        description=(
            "list_buckets returns {bucket_list, continuation} instead of a "
            "bare list / {items, next_token}. Pagination still works."
        ),
        renamed_fields={
            "list_buckets": {
                "items": "bucket_list",
                "next_token": "continuation",
            }
        },
    ),
    "D3": SchemaDiff(
        drift_id="D3",
        description=(
            "region becomes a required argument on list_buckets and "
            "describe_security_groups."
        ),
        required_args={
            "list_buckets": ("region",),
            "describe_security_groups": ("region",),
        },
    ),
    "D4": SchemaDiff(
        drift_id="D4",
        description=(
            "get_bucket_acl is renamed to get_acl. The old name returns an "
            "unknown-tool error that names the new tool."
        ),
        renamed_tools={"get_bucket_acl": "get_acl"},
    ),
}


def get_schema_diff(drift: str | None) -> SchemaDiff | None:
    """Return the schema diff for a drift id, or None for the clean API."""
    if drift is None or str(drift).lower() == "clean":
        return None
    name = str(drift).upper()
    if name not in SCHEMA_DIFFS:
        raise ValueError(f"Unknown drift: {drift}")
    return SCHEMA_DIFFS[name]


def drift_ids() -> tuple[str, ...]:
    return tuple(SCHEMA_DIFFS)


def effective_specs(drift: str | None):
    """Argument specs for the API version a trace actually ran against.

    Validating a drifted run against the clean specs marks every correct call
    as a schema violation. That is exactly what happened to v0_answer2: it read
    `bucket_name` from the tool list, used it 488 times, and the classifier
    called all of it a violation.
    """
    from world.specs import REQUIRED_ARGS, TOOL_ARG_SPECS

    specs = {tool: dict(args) for tool, args in TOOL_ARG_SPECS.items()}
    required = {tool: set(args) for tool, args in REQUIRED_ARGS.items()}

    diff = get_schema_diff(drift)
    if diff is None:
        return specs, required

    for old_tool, new_tool in diff.renamed_tools.items():
        if old_tool in specs:
            specs[new_tool] = specs.pop(old_tool)
            required[new_tool] = required.pop(old_tool, set())

    for tool, pairs in diff.renamed_args.items():
        if tool not in specs:
            continue
        for old, new in pairs.items():
            if old in specs[tool]:
                specs[tool][new] = specs[tool].pop(old)
            if old in required.get(tool, set()):
                required[tool].discard(old)
                required[tool].add(new)

    for tool, names in diff.required_args.items():
        required.setdefault(tool, set()).update(names)

    return specs, required
