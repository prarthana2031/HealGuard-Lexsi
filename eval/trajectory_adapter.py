
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOOL_SCHEMAS = {
    "list_buckets": {
        "required": set(),
        "optional": {"limit", "next_token"},
        "types": {"limit": int, "next_token": str},
    },
    "get_bucket_acl": {
        "required": {"bucket"},
        "optional": set(),
        "types": {"bucket": str},
    },
    "get_bucket_policy": {
        "required": {"bucket"},
        "optional": set(),
        "types": {"bucket": str},
    },
    "get_public_access_block": {
        "required": {"bucket"},
        "optional": set(),
        "types": {"bucket": str},
    },
    "list_iam_roles": {
        "required": set(),
        "optional": {"limit"},
        "types": {"limit": int},
    },
    "get_role_policies": {
        "required": {"role_name"},
        "optional": set(),
        "types": {"role_name": str},
    },
    "describe_security_groups": {
        "required": set(),
        "optional": {"group_ids"},
        "types": {"group_ids": list},
    },
    "finish": {
        "required": {"answer"},
        "optional": set(),
        "types": {"answer": list},
    },
}

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def _extract_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []

    for message in messages:
        if message.get("role") != "assistant":
            continue

        content = message.get("content", "")
        if not isinstance(content, str):
            continue

        for match in TOOL_CALL_PATTERN.finditer(content):
            try:
                call = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                calls.append({
                    "name": None,
                    "arguments": {},
                    "valid": False,
                    "parse_error": str(exc),
                })
                continue

            calls.append(call)

    return calls


def _is_valid_call(call: dict[str, Any]) -> bool:
    name = call.get("name")
    arguments = call.get("arguments")

    if name not in TOOL_SCHEMAS or not isinstance(arguments, dict):
        return False

    schema = TOOL_SCHEMAS[name]
    required = schema["required"]
    optional = schema["optional"]
    types = schema["types"]

    if not required.issubset(arguments):
        return False

    if set(arguments) - required - optional:
        return False

    for key, value in arguments.items():
        expected_type = types.get(key)

        if expected_type is None:
            continue

        if expected_type is int:
            if isinstance(value, bool) or not isinstance(value, int):
                return False
        elif not isinstance(value, expected_type):
            return False

    return True


def adapt_trajectory(raw: dict[str, Any]) -> dict[str, Any]:
    messages = raw.get("messages")

    if not isinstance(messages, list):
        raise ValueError("Trajectory messages must be a list.")

    user_messages = [
        message.get("content", "")
        for message in messages
        if message.get("role") == "user"
    ]

    if not user_messages:
        raise ValueError(f"{raw.get('task_id')}: missing user message")

    calls = _extract_tool_calls(messages)

    valid_calls = sum(_is_valid_call(call) for call in calls)

    signatures = []
    for call in calls:
        if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
            signature = (
                call.get("name"),
                json.dumps(
                    call["arguments"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            signatures.append(signature)

    counts = Counter(signatures)
    redundant_calls = sum(count - 1 for count in counts.values())

    return {
        "id": str(raw["task_id"]),
        "seed": raw.get("seed"),
        "split": raw.get("split"),
        "task": user_messages[-1],
        "messages": messages,
        "tool_calls": calls,
        "reward": 1.0 if raw.get("is_correct") is True else 0.0,
        "success": raw.get("is_correct"),
        "num_steps": len(calls),
        "template_id": "custom_tool_call_v1",
        "error": None,
        "valid_tool_calls": valid_calls,
        "total_tool_calls": len(calls),
        "redundant_calls": redundant_calls,
        "finished": any(
            call.get("name") == "finish"
            for call in calls
            if isinstance(call, dict)
        ),
        "final_answer": raw.get("final_answer"),
    }


def adapt_jsonl(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with input_path.open("r", encoding="utf-8") as source:
        with output_path.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                raw = json.loads(line)

                try:
                    adapted = adapt_trajectory(raw)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid trajectory on line {line_number}: {exc}"
                    ) from exc

                target.write(json.dumps(adapted) + "\n")
                count += 1

    if count == 0:
        raise ValueError("Input dataset is empty.")

    return count
