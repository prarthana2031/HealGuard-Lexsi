import pytest

from eval.trajectory_adapter import adapt_trajectory


def make_trajectory(**overrides):
    trajectory = {
        "task_id": "test_001",
        "seed": 42,
        "split": "drift_targeted",
        "family": "F1",
        "is_correct": False,
        "messages": [
            {"role": "user", "content": "Inspect cloud resources."},
            {
                "role": "assistant",
                "content": (
                    '<tool_call>{"name":"list_buckets",'
                    '"arguments":{"limit":5}}</tool_call>'
                ),
            },
        ],
        "final_answer": [],
    }
    trajectory.update(overrides)
    return trajectory


def test_preserves_gate_metadata():
    result = adapt_trajectory(make_trajectory())

    assert result["id"] == "test_001"
    assert result["split"] == "drift_targeted"
    assert result["family"] == "F1"
    assert result["success"] is False
    assert result["failure_mode"] == "residual"


def test_preserves_task_family_fallback():
    result = adapt_trajectory(
        make_trajectory(family=None, task_family="F2")
    )

    assert result["family"] == "F2"


def test_preserves_existing_failure_mode():
    result = adapt_trajectory(
        make_trajectory(failure_mode="schema_violation")
    )

    assert result["failure_mode"] == "schema_violation"


def test_successful_trajectory_has_no_failure_mode():
    result = adapt_trajectory(make_trajectory(is_correct=True))

    assert result["success"] is True
    assert result["failure_mode"] is None


def test_extracts_tool_call_counts():
    result = adapt_trajectory(make_trajectory())

    assert result["total_tool_calls"] == 1
    assert result["valid_tool_calls"] == 1
    assert result["num_steps"] == 1