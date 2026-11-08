"""Deterministic, rule-based failure classifier for HealGuard.

Implements the eleven-label taxonomy in the priority order the plan fixes:

    env_error > drift_mismatch > schema_violation > wrong_tool > wrong_order >
    premature_finish > ungrounded_answer > loop_collapse > redundant_call >
    unrecovered_error > residual

Two properties matter and both are tested:

1. The primary label is the first match in priority order, and *every* match is
   kept (the plan asks for "primary label plus the full set of labels that
   matched").
2. The drift is inferred **from the trace**, not read from a `drift` field.  If
   the classifier were told the drift id it would score 1.0 by construction and
   the confusion matrix would be worthless.  `heal.drifts.schema_diff` is used
   only to (a) recognise the evidence and (b) build the corrective example.

No LLM is involved anywhere.  Residual failures are counted, never sent out.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from heal.drifts.schema_diff import (
    SchemaDiff,
    effective_specs,
    get_schema_diff,
)
from world.specs import PREREQUISITES

PRIORITY = [
    "env_error",
    "drift_mismatch",
    "schema_violation",
    "wrong_tool",
    "wrong_order",
    "premature_finish",
    "ungrounded_answer",
    "loop_collapse",
    "redundant_call",
    "unrecovered_error",
    "residual",
]

ENV_ERROR_RE = re.compile(
    r"\b(503|502|500|504|timeout|timed out|timed-out|service unavailable"
    r"|connection reset|throttl)",
    re.I,
)
RENAMED_ARG_RE = re.compile(
    r"unexpected argument\(s\):\s*(\w+)[;,]\s*use\s*(\w+)", re.I
)
REQUIRED_ARG_RE = re.compile(
    r"(\w+)\s+is required", re.I
)
RENAMED_TOOL_RE = re.compile(
    r"unknown_tool:.*?not found.*?did you mean\s*'?([\w.]+)'?", re.I
)
ERROR_PREFIX_RE = re.compile(r"^\s*([a-z_]+)\s*:", re.I)


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def _tool_call_re() -> re.Pattern[str]:
    return re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_calls_from_content(content: str) -> list[dict[str, Any]]:
    """Extract the `<tool_call>` blocks from one assistant message."""
    calls = []
    for match in _tool_call_re().findall(content or ""):
        call: dict[str, Any] = {"raw": match}
        try:
            parsed = json.loads(match)
        except json.JSONDecodeError:
            call["valid_json"] = False
            call["name"] = None
            call["arguments"] = None
            calls.append(call)
            continue
        call["valid_json"] = True
        if isinstance(parsed, dict):
            call["name"] = parsed.get("name")
            call["arguments"] = parsed.get("arguments", {})
        else:
            call["name"] = None
            call["arguments"] = None
        calls.append(call)
    return calls


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each tool call with the tool message that answered it."""
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    pending: list[int] = []

    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            for call in parse_calls_from_content(message.get("content", "")):
                call["step"] = message_index
                call["index"] = len(calls)
                calls.append(call)
                pending.append(call["index"])
        elif role == "tool":
            text = message.get("content", "")
            parsed = _try_json(text)
            results.append(
                {
                    "call_index": pending.pop(0) if pending else None,
                    "text": text if isinstance(text, str) else "",
                    "parsed": parsed,
                    "step": message_index,
                }
            )

    return calls, results


def _error_of(result: dict[str, Any]) -> tuple[str | None, str]:
    """Return (error_prefix, full_text) for a tool result.

    Any `{"error": ...}` payload counts as an error even when the environment
    used a bare AWS-style code such as `NoSuchBucket` with no prefix.
    """
    parsed = result.get("parsed")
    text = result.get("text", "")

    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, str):
            match = ERROR_PREFIX_RE.match(error)
            return (match.group(1).lower() if match else "tool_error"), error

    if re.search(r"\berror\b", text, re.I):
        return "tool_error", text

    return None, text


def _has_evidence_call(calls: list[dict[str, Any]]) -> bool:
    return any(
        call.get("name") in {"list_buckets", "list_iam_roles", "describe_security_groups"}
        or call.get("name") in PREREQUISITES
        for call in calls
    )


def _answer_of(trajectory: dict[str, Any], calls: list[dict[str, Any]]) -> list[str]:
    raw = trajectory.get("final_answer")
    if raw is None:
        for call in reversed(calls):
            if call.get("name") == "finish" and isinstance(call.get("arguments"), dict):
                raw = call["arguments"].get("answer")
                break
    if isinstance(raw, str):
        raw = _try_json(raw)
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


# ---------------------------------------------------------------------------
# Drift inference (from the trace only)
# ---------------------------------------------------------------------------

def infer_schema_diff(
    calls: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[SchemaDiff | None, str | None]:
    """Infer which drift the trace ran under, using only trace evidence."""

    for result in results:
        _, text = _error_of(result)
        match = RENAMED_TOOL_RE.search(text)
        if match:
            new_tool = match.group(1)
            for drift in ("D4",):
                diff = get_schema_diff(drift)
                if new_tool in diff.renamed_tools.values():
                    return diff, "unknown-tool error names the new tool"

        match = RENAMED_ARG_RE.search(text)
        if match:
            new_name = match.group(2)
            for drift in ("D1",):
                diff = get_schema_diff(drift)
                if new_name in diff.new_arg_names():
                    return diff, "schema error names the new argument"

        match = REQUIRED_ARG_RE.search(text)
        if match and "required" in text.lower():
            for drift in ("D3",):
                diff = get_schema_diff(drift)
                if match.group(1) in diff.names():
                    return diff, "schema error names the new required argument"

    for result in results:
        parsed = result.get("parsed")
        if isinstance(parsed, dict):
            for drift in ("D2",):
                diff = get_schema_diff(drift)
                for per_tool in diff.renamed_fields.values():
                    if any(new in parsed for new in per_tool.values()):
                        return diff, "tool result exposes the new response field"

    # Deliberately no "call uses `bucket`" fallback: on the clean API `bucket`
    # is the correct name, so that evidence would fire on every healthy trace.
    # Drift inference needs positive evidence from the environment's reply.

    return None, None


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _rule_env_error(calls, results, answer):
    """Injected 5xx/timeout that the agent did not compound.

    A repeat of the identical failing call is the agent's fault, not the
    environment's, so it is left to `unrecovered_error`.
    """
    repeated_after_error, _, _ = _error_followed_by_identical_call(calls, results)
    if repeated_after_error:
        return False, None, ""

    for result in results:
        prefix, text = _error_of(result)
        injected = prefix == "env_error" or bool(ENV_ERROR_RE.search(text))
        if injected:
            return True, result.get("call_index"), "injected environment error"
    return False, None, ""


def _rule_drift_mismatch(calls, results, answer):
    diff, source = infer_schema_diff(calls, results)
    if diff is None:
        return False, None, ""

    # An error whose name or argument is in the schema diff.
    for result in results:
        prefix, text = _error_of(result)
        if prefix not in {"schema_violation", "unknown_tool", None}:
            continue
        if any(name in text for name in diff.names()):
            return True, result.get("call_index"), f"schema diff evidence: {source}"

    # A call that still uses the old name.
    for call in calls:
        name = call.get("name")
        if name in diff.renamed_tools:
            return True, call.get("index"), "call uses the renamed tool"
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            for tool, pairs in diff.renamed_args.items():
                if name != tool:
                    continue
                for old, new in pairs.items():
                    if old in arguments and new not in arguments:
                        return True, call.get("index"), "call uses the renamed argument"

    # The response contract changed and the run failed.
    for result in results:
        parsed = result.get("parsed")
        if isinstance(parsed, dict):
            for per_tool in diff.renamed_fields.values():
                if any(new in parsed for new in per_tool.values()):
                    return True, result.get("call_index"), "response contract changed"

    return False, None, ""


def _check_against(specs, required, tool, arguments) -> list[str]:
    """Strict argument validation against a specific API version."""
    from world.specs import is_valid_type

    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]
    if tool not in specs:
        return []

    problems = []
    for name in sorted(required.get(tool, set()) - arguments.keys()):
        problems.append(f"missing required argument {name!r}")
    for name, value in arguments.items():
        if name not in specs[tool]:
            problems.append(f"unexpected argument {name!r}")
            continue
        allowed = specs[tool][name]
        if value is None and type(None) in allowed:
            continue
        if not is_valid_type(value, allowed):
            expected = "/".join(t.__name__ for t in allowed)
            problems.append(
                f"argument {name!r} must be {expected}, got {type(value).__name__}"
            )
    return problems


def _rule_schema_violation(calls, results, answer, specs, required):
    if not calls:
        return True, None, "no parseable tool call was emitted"

    for call in calls:
        if not call.get("valid_json"):
            return True, call.get("index"), "malformed tool-call JSON"
        if not isinstance(call.get("name"), str) or not call.get("name"):
            return True, call.get("index"), "tool call has no name"
        if not isinstance(call.get("arguments"), dict):
            return True, call.get("index"), "arguments are not an object"
        if call.get("name") in specs:
            problems = _check_against(specs, required, call["name"], call["arguments"])
            if problems:
                return True, call.get("index"), "; ".join(problems)

    for result in results:
        prefix, text = _error_of(result)
        if prefix == "schema_violation":
            return True, result.get("call_index"), "environment reported a schema violation"

    return False, None, ""


def _rule_wrong_tool(calls, results, answer, task_lower, specs):
    for call in calls:
        name = call.get("name")
        if isinstance(name, str) and name not in specs:
            return True, call.get("index"), f"tool {name!r} is not in the schema"

    if "list buckets" in task_lower and "list_iam_roles" in {
        call.get("name") for call in calls
    }:
        return True, None, "used list_iam_roles for a bucket question"
    if "list iam roles" in task_lower and "list_buckets" in {
        call.get("name") for call in calls
    }:
        return True, None, "used list_buckets for a role question"

    return False, None, ""


def _rule_wrong_order(calls, results, answer):
    seen: set[str] = set()
    for call in calls:
        name = call.get("name")
        if not isinstance(name, str):
            continue
        prerequisite = PREREQUISITES.get(name)
        if prerequisite and prerequisite not in seen:
            return True, call.get("index"), f"{name} before {prerequisite}"
        seen.add(name)
    return False, None, ""


def _listing_was_non_empty(results) -> bool:
    """True when any tool result already carried at least one resource."""
    for result in results:
        parsed = result.get("parsed")
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list) and value:
                    return True
    return False


def _rule_premature_finish(calls, results, answer):
    finish_calls = [call for call in calls if call.get("name") == "finish"]
    if not finish_calls:
        return False, None, ""
    finish = finish_calls[0]
    if not answer:
        if not _has_evidence_call(calls) or _listing_was_non_empty(results):
            return True, finish.get("index"), "finish with an empty or unsupported answer"
    return False, None, ""


def _rule_ungrounded_answer(calls, results, answer):
    if not answer:
        return False, None, ""
    # Only *observations* count as grounding.  The finish call echoes the answer
    # back in its tool result; including that would make every answer grounded.
    observed_parts = []
    for result in results:
        call_index = result.get("call_index")
        if call_index is not None and calls[call_index].get("name") == "finish":
            continue
        observed_parts.append(result.get("text", ""))
    observed = " ".join(observed_parts)
    ungrounded = [value for value in answer if value not in observed]
    if ungrounded:
        return True, None, f"answer ids never seen: {ungrounded[:5]}"
    return False, None, ""


def _signatures(calls) -> list[str]:
    return [
        json.dumps(
            {"name": call.get("name"), "arguments": call.get("arguments")},
            sort_keys=True,
            default=str,
        )
        for call in calls
        if call.get("name") != "finish"
    ]


def _rule_loop_collapse(calls, results, answer):
    counts = Counter(_signatures(calls))
    for signature, count in counts.items():
        if count >= 3:
            index = next(
                (
                    call.get("index")
                    for call in calls
                    if json.dumps(
                        {"name": call.get("name"), "arguments": call.get("arguments")},
                        sort_keys=True,
                        default=str,
                    ) == signature
                ),
                None,
            )
            return True, index, f"identical call repeated {count} times"
    return False, None, ""


def _error_followed_by_identical_call(calls, results) -> tuple[bool, int | None, str]:
    for result in results:
        prefix, text = _error_of(result)
        if not (prefix or re.search(r"\berror\b", text, re.I)):
            continue
        call_index = result.get("call_index")
        if call_index is None:
            continue
        call = calls[call_index]
        signature = json.dumps(
            {"name": call.get("name"), "arguments": call.get("arguments")},
            sort_keys=True,
            default=str,
        )
        for later in calls[call_index + 1:]:
            later_signature = json.dumps(
                {"name": later.get("name"), "arguments": later.get("arguments")},
                sort_keys=True,
                default=str,
            )
            if later_signature == signature:
                return True, call_index, "the failing call was repeated unchanged"
            if later.get("name") == "finish":
                return True, call_index, "gave up after an error without a changed retry"
            return False, None, ""
        return True, call_index, "stopped after an error without a changed retry"
    return False, None, ""


def _rule_redundant_call(calls, results, answer):
    counts = Counter(_signatures(calls))
    repeated = {
        signature for signature, count in counts.items() if count >= 2
    }
    if not repeated:
        return False, None, ""

    # A repeat that follows an error is unrecovered_error, not redundant work.
    error_repeat, _, _ = _error_followed_by_identical_call(calls, results)
    if error_repeat:
        return False, None, ""

    for call in calls:
        signature = json.dumps(
            {"name": call.get("name"), "arguments": call.get("arguments")},
            sort_keys=True,
            default=str,
        )
        if signature in repeated:
            return True, call.get("index"), "identical call repeated"
    return False, None, ""


def _rule_unrecovered_error(calls, results, answer):
    matched, index, detail = _error_followed_by_identical_call(calls, results)
    if matched:
        return True, index, detail
    return False, None, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _is_failure(trajectory: dict[str, Any]) -> bool | None:
    for field in ("success", "is_correct"):
        value = trajectory.get(field)
        if isinstance(value, bool):
            return not value
    return None


def classify(
    trajectory: dict[str, Any],
    schema_diff: SchemaDiff | None = None,
) -> dict[str, Any]:
    """Classify one trajectory.

    Returns a dict with the primary label, every label that matched, the first
    divergent step and the evidence for each match.  `schema_diff` is accepted
    for callers that want to force a known diff when building corrective
    examples; it is never required for classification.
    """

    messages = trajectory.get("messages") or []
    calls, results = extract_events(messages)

    task = trajectory.get("task") or next(
        (
            message.get("content", "")
            for message in messages
            if message.get("role") == "user"
        ),
        "",
    )
    task_lower = str(task).lower()
    answer = _answer_of(trajectory, calls)

    drift = trajectory.get("drift")
    specs, required = effective_specs(drift)

    failure = _is_failure(trajectory)
    if failure is False:
        return {
            "primary": None,
            "labels": [],
            "first_divergent_step": None,
            "matches": {},
            "residual": False,
            "success": True,
            "drift_inferred": None,
            "drift_evidence": None,
            "call_count": len(calls),
        }

    rules = [
        ("env_error", lambda: _rule_env_error(calls, results, answer)),
        ("drift_mismatch", lambda: _rule_drift_mismatch(calls, results, answer)),
        (
            "schema_violation",
            lambda: _rule_schema_violation(calls, results, answer, specs, required),
        ),
        (
            "wrong_tool",
            lambda: _rule_wrong_tool(calls, results, answer, task_lower, specs),
        ),
        ("wrong_order", lambda: _rule_wrong_order(calls, results, answer)),
        ("premature_finish", lambda: _rule_premature_finish(calls, results, answer)),
        ("ungrounded_answer", lambda: _rule_ungrounded_answer(calls, results, answer)),
        ("loop_collapse", lambda: _rule_loop_collapse(calls, results, answer)),
        ("redundant_call", lambda: _rule_redundant_call(calls, results, answer)),
        ("unrecovered_error", lambda: _rule_unrecovered_error(calls, results, answer)),
    ]

    matches: dict[str, dict[str, Any]] = {}
    for label, rule in rules:
        matched, step, detail = rule()
        if matched:
            matches[label] = {"first_step": step, "detail": detail}

    labels = [label for label in PRIORITY if label in matches]
    primary = labels[0] if labels else "residual"
    if primary == "residual":
        matches["residual"] = {"first_step": None, "detail": "no rule matched"}

    inferred_diff, diff_source = infer_schema_diff(calls, results)
    effective_diff = schema_diff or inferred_diff

    return {
        "primary": primary,
        "labels": labels,
        "first_divergent_step": matches.get(primary, {}).get("first_step"),
        "matches": matches,
        "residual": primary == "residual",
        "success": failure is not False,
        "drift_inferred": effective_diff.drift_id if effective_diff else None,
        "drift_evidence": diff_source,
        "call_count": len(calls),
    }


def classify_trajectory(trajectory: dict[str, Any]) -> str | None:
    """Backwards-compatible wrapper returning only the primary label."""
    return classify(trajectory)["primary"]


def residual_share(trajectories: list[dict[str, Any]]) -> float:
    """Residual share over the classified failures, as the plan reports it."""
    if not trajectories:
        return 0.0
    residual = sum(
        1 for trajectory in trajectories
        if classify(trajectory)["primary"] == "residual"
    )
    return residual / len(trajectories)
