
import json
import re
from collections import Counter


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


def extract_tool_calls(messages):
    """Extract tool calls from assistant messages."""
    calls = []

    for message in messages:
        if message.get("role") != "assistant":
            continue

        content = message.get("content", "")

        matches = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            content,
            re.S,
        )

        for match in matches:
            try:
                calls.append(json.loads(match))
            except json.JSONDecodeError:
                calls.append({"_invalid_json": True})

    return calls


def classify_trajectory(trajectory):
    """Assign exactly one primary failure label."""

    messages = trajectory.get("messages", [])
    calls = extract_tool_calls(messages)

    tool_names = [
        call.get("name")
        for call in calls
        if isinstance(call, dict) and call.get("name")
    ]

    tool_messages = [
        message.get("content", "")
        for message in messages
        if message.get("role") == "tool"
    ]

    if (
        trajectory.get("is_correct") is True
        or trajectory.get("success") is True
    ):
        return None

    task = trajectory.get("task", "")

    if not task:
        task = next(
            (
                message.get("content", "")
                for message in messages
                if message.get("role") == "user"
            ),
            "",
        )

    task_lower = task.lower()

    # 1. Explicit environment or execution errors.
    if any(
        re.search(
            r"\b(environment error|execution environment error)\b",
            text,
            re.I,
        )
        for text in tool_messages
    ):
        return "env_error"

    # 2. Drift mismatch.
    # Requires environment/tool configuration evidence.
    # Do not infer it from trajectory filenames.

    # 3. Schema violation.
    if any(call.get("_invalid_json") for call in calls):
        return "schema_violation"

    for call in calls:
        arguments = call.get("arguments", {})

        if not isinstance(arguments, dict):
            return "schema_violation"

        if (
            "limit" in arguments
            and (
                not isinstance(arguments["limit"], int)
                or isinstance(arguments["limit"], bool)
            )
        ):
            return "schema_violation"

        if (
            call.get("name") == "get_bucket_policy"
            and "bucket" not in arguments
        ):
            return "schema_violation"

    # 4. Wrong tool.
    if (
        "list buckets" in task_lower
        and "list_iam_roles" in tool_names
    ):
        return "wrong_tool"

    if (
        "list iam roles" in task_lower
        and "list_buckets" in tool_names
    ):
        return "wrong_tool"

    # 5. Wrong order.
    if (
        "public buckets" in task_lower
        and "get_bucket_policy" in tool_names
        and "list_buckets" not in tool_names
    ):
        return "wrong_order"

    # 6. Premature finish.
    if (
        "list buckets" in task_lower
        and "finish" in tool_names
    ):
        final_answer = trajectory.get("final_answer", "")

        try:
            answer = json.loads(final_answer)
        except (json.JSONDecodeError, TypeError):
            answer = None

        if "list_buckets" not in tool_names or answer == []:
            return "premature_finish"

    # 7. Ungrounded answer.
    observed_text = " ".join(tool_messages)
    final_answer = trajectory.get("final_answer", "")

    if final_answer:
        try:
            answer_values = json.loads(final_answer)

            if isinstance(answer_values, list):
                if any(
                    str(value) not in observed_text
                    for value in answer_values
                ):
                    return "ungrounded_answer"

        except (json.JSONDecodeError, TypeError):
            pass

    # Build normalized tool-call signatures.
    signatures = [
        json.dumps(call, sort_keys=True)
        for call in calls
        if isinstance(call, dict)
        and "name" in call
        and "finish" != call.get("name")
    ]

    counts = Counter(signatures)

    # 8. Loop collapse.
    if any(count >= 3 for count in counts.values()):
        return "loop_collapse"

    # 9. Redundant call.
    if any(count >= 2 for count in counts.values()):
        return "redundant_call"

    # 10. Unrecovered error.
    has_tool_error = any(
        re.search(
            r"\b(error|timeout|accessdenied)\b",
            text,
            re.I,
        )
        for text in tool_messages
    )

    if has_tool_error:
        return "unrecovered_error"

    # 11. Residual fallback.
    return "residual"
