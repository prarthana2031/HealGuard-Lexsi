import json
from classifier.rules import classify_trajectory

def call(name, args):
    return "<tool_call>\n" + json.dumps({
        "name": name, "arguments": args
    }) + "\n</tool_call>"

def test_unrecovered_error():
    trajectory = {
        "task": "Check policy.",
        "is_correct": False,
        "final_answer": "[]",
        "messages": [
            {"role": "user", "content": "Check policy."},
            {"role": "assistant", "content": call(
                "get_bucket_policy", {"bucket": "b1"}
            )},
            {"role": "tool", "content": "Error: AccessDenied"},
            {"role": "assistant", "content": "Access was denied."},
        ],
    }
    assert classify_trajectory(trajectory) == "unrecovered_error"

def test_residual():
    trajectory = {
        "task": "Review cloud posture.",
        "is_correct": False,
        "final_answer": "[]",
        "messages": [
            {"role": "user", "content": "Review cloud posture."},
            {"role": "assistant", "content": "I cannot complete the review."},
        ],
    }
    assert classify_trajectory(trajectory) == "residual"
