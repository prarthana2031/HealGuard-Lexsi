
"""HealGuard evaluation metrics using AuditKIT 1.1.3.

Trajectory input follows agent/trajectory_schema.md.
"""

from __future__ import annotations

import math
from typing import Any

import auditkit as ak


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

LOOP_DETECTOR = ak.AgentLoopDetection(
    repetition_threshold=3,
    similarity_threshold=0.85,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _get_trajectory(
    sample: ak.Sample,
    context: Any = None,
) -> dict[str, Any]:
    """Retrieve the original HealGuard trajectory."""

    if isinstance(context, dict):
        trajectory = context.get("trajectory")
        if isinstance(trajectory, dict):
            return trajectory

    metadata = getattr(sample, "metadata", {}) or {}
    trajectory = metadata.get("trajectory")

    if isinstance(trajectory, dict):
        return trajectory

    raise ValueError(
        "HealGuard trajectory not found. "
        "Pass it through sample.metadata['trajectory'] "
        "or context['trajectory']."
    )


def _score(
    name: str,
    value: float,
    direction: ak.Direction,
    **metadata: Any,
) -> ak.Score:
    """Create a consistently formatted AuditKIT score."""

    return ak.Score(
        name=name,
        value=value,
        direction=direction,
        metadata=metadata,
    )


def _not_applicable(
    name: str,
    direction: ak.Direction,
    reason: str,
) -> ak.Score:
    """Represent an undefined metric without treating it as zero."""

    return _score(
        name,
        math.nan,
        direction,
        not_applicable=True,
        reason=reason,
    )


def _validate_count(
    trajectory: dict[str, Any],
    field: str,
) -> int:
    """Read and validate a non-negative integer count."""

    value = trajectory.get(field)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Trajectory field '{field}' must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"Trajectory field '{field}' cannot be negative."
        )

    return value


# ---------------------------------------------------------
# 1. Task Success
# Direction: MAXIMIZE
# ---------------------------------------------------------

@ak.scorer(
    direction=ak.Direction.MAXIMIZE,
    name="task_success",
)
def task_success(
    sample: ak.Sample,
    output: str,
    context: Any = None,
) -> ak.Score:

    trajectory = _get_trajectory(sample, context)

    success = trajectory.get("success")

    if not isinstance(success, bool):
        raise ValueError(
            "Trajectory field 'success' must be boolean."
        )

    return _score(
        "task_success",
        float(success),
        ak.Direction.MAXIMIZE,
        success=success,
    )


# ---------------------------------------------------------
# 2. Valid Call Rate
# Direction: MAXIMIZE
# ---------------------------------------------------------

@ak.scorer(
    direction=ak.Direction.MAXIMIZE,
    name="valid_call_rate",
)
def valid_call_rate(
    sample: ak.Sample,
    output: str,
    context: Any = None,
) -> ak.Score:

    trajectory = _get_trajectory(sample, context)

    valid = _validate_count(trajectory, "valid_tool_calls")
    total = _validate_count(trajectory, "total_tool_calls")

    if valid > total:
        raise ValueError(
            "valid_tool_calls cannot exceed total_tool_calls."
        )

    if total == 0:
        return _not_applicable(
            "valid_call_rate",
            ak.Direction.MAXIMIZE,
            "No tool calls were attempted.",
        )

    return _score(
        "valid_call_rate",
        valid / total,
        ak.Direction.MAXIMIZE,
        valid_tool_calls=valid,
        total_tool_calls=total,
    )


# ---------------------------------------------------------
# 3. Redundant Rate
# Direction: MINIMIZE
# ---------------------------------------------------------

@ak.scorer(
    direction=ak.Direction.MINIMIZE,
    name="redundant_rate",
)
def redundant_rate(
    sample: ak.Sample,
    output: str,
    context: Any = None,
) -> ak.Score:

    trajectory = _get_trajectory(sample, context)

    redundant = _validate_count(trajectory, "redundant_calls")
    total = _validate_count(trajectory, "total_tool_calls")

    if redundant > total:
        raise ValueError(
            "redundant_calls cannot exceed total_tool_calls."
        )

    if total == 0:
        return _not_applicable(
            "redundant_rate",
            ak.Direction.MINIMIZE,
            "No tool calls were attempted.",
        )

    return _score(
        "redundant_rate",
        redundant / total,
        ak.Direction.MINIMIZE,
        redundant_calls=redundant,
        total_tool_calls=total,
    )


# ---------------------------------------------------------
# 4. Loop Rate
# Direction: MINIMIZE
# Uses AuditKIT AgentLoopDetection
# ---------------------------------------------------------

@ak.scorer(
    direction=ak.Direction.MINIMIZE,
    name="loop_rate",
)
def loop_rate(
    sample: ak.Sample,
    output: str,
    context: Any = None,
) -> ak.Score:

    trajectory = _get_trajectory(sample, context)

    messages = trajectory.get("messages")
    calls = trajectory.get("tool_calls")

    if not isinstance(messages, list):
        raise ValueError(
            "Trajectory field 'messages' must be a list."
        )

    if not isinstance(calls, list):
        raise ValueError(
            "Trajectory field 'tool_calls' must be a list."
        )

    # The trajectory contract stores calls in a flat ordered list.
    # Group them into one turn for AuditKIT's structured trace.
    trace = {
        "messages": messages,
        "tool_calls": [calls] if calls else [],
    }

    detector_context = {"trace": trace}

    detector_scores = LOOP_DETECTOR.score(
        sample,
        output,
        context=detector_context,
    )

    if not detector_scores:
        return _not_applicable(
            "loop_rate",
            ak.Direction.MINIMIZE,
            "Insufficient assistant-message or tool-call data.",
        )

    detector_score = detector_scores[0]
    looped = detector_score.metadata.get("looped")

    if not isinstance(looped, bool):
        raise ValueError(
            "AuditKIT loop detector did not return "
            "a valid boolean 'looped' result."
        )

    return _score(
        "loop_rate",
        float(looped),
        ak.Direction.MINIMIZE,
        loop_detected=looped,
        detector_value=detector_score.value,
        detector_reason=detector_score.reason,
        detector_metadata=detector_score.metadata,
    )


# ---------------------------------------------------------
# 5. Steps on Success
# Direction: MINIMIZE
# ---------------------------------------------------------

@ak.scorer(
    direction=ak.Direction.MINIMIZE,
    name="steps_on_success",
)
def steps_on_success(
    sample: ak.Sample,
    output: str,
    context: Any = None,
) -> ak.Score:

    trajectory = _get_trajectory(sample, context)

    success = trajectory.get("success")

    if not isinstance(success, bool):
        raise ValueError(
            "Trajectory field 'success' must be boolean."
        )

    if not success:
        return _not_applicable(
            "steps_on_success",
            ak.Direction.MINIMIZE,
            "Trajectory was unsuccessful.",
        )

    steps = _validate_count(trajectory, "num_steps")

    return _score(
        "steps_on_success",
        float(steps),
        ak.Direction.MINIMIZE,
        num_steps=steps,
    )


# ---------------------------------------------------------
# Trajectory Adapter
# ---------------------------------------------------------

def trajectory_to_sample(
    trajectory: dict[str, Any],
) -> tuple[ak.Sample, str, dict[str, Any]]:
    """Convert a HealGuard trajectory into an AuditKIT sample.

    Returns:
        sample: AuditKIT Sample
        output: Final assistant output
        context: Context containing the original trajectory
    """

    required_fields = [
        "id",
        "task",
        "messages",
        "tool_calls",
        "success",
        "valid_tool_calls",
        "total_tool_calls",
        "redundant_calls",
        "num_steps",
    ]

    missing = [
        field for field in required_fields
        if field not in trajectory
    ]

    if missing:
        raise ValueError(
            f"Missing required trajectory fields: {missing}"
        )

    messages = trajectory["messages"]

    if not isinstance(messages, list):
        raise ValueError("'messages' must be a list.")

    # Extract the last assistant text as the model output.
    output = ""

    for message in reversed(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ):
            output = message["content"]
            break

    sample = ak.Sample(
        id=str(trajectory["id"]),
        input=str(trajectory["task"]),
        actual_output=output,
        actual_trace={
            "messages": messages,
            "tool_calls": (
                [trajectory["tool_calls"]]
                if trajectory["tool_calls"]
                else []
            ),
        },
        metadata={
            "trajectory": trajectory,
        },
    )

    context = {
        "trajectory": trajectory,
    }

    return sample, output, context


# ---------------------------------------------------------
# Export all five scorers
# ---------------------------------------------------------

METRICS = [
    task_success,
    valid_call_rate,
    redundant_rate,
    loop_rate,
    steps_on_success,
]
