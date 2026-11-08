"""Focused unit tests for individual classifier rules."""

import json

from classifier.rules import classify_trajectory


def call(name, args):
    return "<tool_call>\n" + json.dumps(
        {"name": name, "arguments": args}
    ) + "\n</tool_call>"


def test_unrecovered_error():
    """An error followed by the identical call, with every prerequisite met."""
    trajectory = {
        "task": "Check ACL.",
        "is_correct": False,
        "final_answer": ["bucket-1"],
        "messages": [
            {"role": "user", "content": "Check ACL."},
            {"role": "assistant", "content": call("list_buckets", {})},
            {"role": "tool", "content": json.dumps({"items": ["bucket-1"]})},
            {"role": "assistant", "content": call("get_bucket_acl", {"bucket": "bucket-1"})},
            {"role": "tool", "content": json.dumps({"error": "NoSuchBucket"})},
            {"role": "assistant", "content": call("get_bucket_acl", {"bucket": "bucket-1"})},
            {"role": "tool", "content": json.dumps({"error": "NoSuchBucket"})},
            {"role": "assistant", "content": call("finish", {"answer": ["bucket-1"]})},
            {"role": "tool", "content": json.dumps({"status": "success"})},
        ],
    }
    assert classify_trajectory(trajectory) == "unrecovered_error"


def test_genuine_residual():
    """Valid calls, grounded answer, no repeat, no error — just incomplete."""
    trajectory = {
        "task": "Review cloud posture.",
        "is_correct": False,
        "final_answer": ["bucket-2"],
        "messages": [
            {"role": "user", "content": "Review cloud posture."},
            {"role": "assistant", "content": call("list_buckets", {})},
            {"role": "tool", "content": json.dumps({"items": ["bucket-1", "bucket-2"]})},
            {"role": "assistant", "content": call("finish", {"answer": ["bucket-2"]})},
            {"role": "tool", "content": json.dumps({"status": "success"})},
        ],
    }
    assert classify_trajectory(trajectory) == "residual"


def test_finish_result_does_not_ground_the_answer():
    """The finish echo must not be treated as an observation."""
    trajectory = {
        "task": "Review cloud posture.",
        "is_correct": False,
        "final_answer": ["bucket-that-never-existed"],
        "messages": [
            {"role": "user", "content": "Review cloud posture."},
            {"role": "assistant", "content": call("list_buckets", {})},
            {"role": "tool", "content": json.dumps({"items": ["bucket-1"]})},
            {
                "role": "assistant",
                "content": call("finish", {"answer": ["bucket-that-never-existed"]}),
            },
            {
                "role": "tool",
                "content": json.dumps({"answer": ["bucket-that-never-existed"]}),
            },
        ],
    }
    assert classify_trajectory(trajectory) == "ungrounded_answer"


def _bucket_trace(argument_name, drift):
    """One bucket read using `argument_name`, run under `drift`."""
    return {
        "task": "Find all vulnerable cloud resources.",
        "drift": drift,
        "is_correct": False,
        "final_answer": ["bucket-1"],
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": call("list_buckets", {})},
            {"role": "tool", "content": json.dumps({"items": ["bucket-1", "bucket-2"]})},
            {"role": "assistant", "content": call("get_bucket_acl", {argument_name: "bucket-1"})},
            {"role": "tool", "content": json.dumps({"acl_public": True})},
            {"role": "assistant", "content": call("finish", {"answer": ["bucket-1"]})},
            {"role": "tool", "content": json.dumps({"status": "success"})},
        ],
    }


def test_drifted_argument_name_is_valid_under_the_drift():
    """v0_answer2 read `bucket_name` off the tool list; that is not a violation."""
    assert classify_trajectory(_bucket_trace("bucket_name", "D1")) != "schema_violation"


def test_the_same_call_is_a_violation_on_the_clean_api():
    """The check must still fire when the schema really does say `bucket`."""
    assert classify_trajectory(_bucket_trace("bucket_name", None)) == "schema_violation"
