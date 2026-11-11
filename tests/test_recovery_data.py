"""Lock the recovery invariant that v1_sft violated.

The defect: `data/heal/sft_chosen.jsonl` pairs a clean system prompt
(`bucket`) with assistant calls using `bucket_name`, and contains no error
turns.  The model learned schema override instead of recovery, and on clean
traffic that leaked as a 38% schema-violation rate.

These tests fail if that defect ever comes back, and they fail if the generated
recovery data stops being a *conditional* correction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from heal.recovery import check_recovery_invariant, clean_violations

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "data" / "heal" / "recovery_sft.jsonl"
SFT_CHOSEN = ROOT / "data" / "heal" / "sft_chosen.jsonl"


def _row(*messages: dict) -> list[dict]:
    return list(messages)


def _call(name: str, **arguments) -> dict:
    return {
        "role": "assistant",
        "content": "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>",
    }


def _error(text: str) -> dict:
    return {"role": "tool", "content": json.dumps({"error": text})}


def _ok(payload: dict) -> dict:
    return {"role": "tool", "content": json.dumps(payload)}


def test_conditional_correction_passes():
    messages = _row(
        {"role": "system", "content": "Tools: get_bucket_acl(bucket)"},
        {"role": "user", "content": "Find all vulnerable cloud resources."},
        _call("get_bucket_acl", bucket="b-1"),
        _error("schema_violation: unexpected argument(s): bucket; use bucket_name"),
        _call("get_bucket_acl", bucket_name="b-1"),
        _ok({"acl_public": False}),
    )
    assert check_recovery_invariant(messages, "D1") == []


def test_schema_override_without_an_error_fails():
    """This is the exact v1_sft pattern: bucket_name with no error before it."""
    messages = _row(
        {"role": "system", "content": "Tools: get_bucket_acl(bucket)"},
        {"role": "user", "content": "Find all vulnerable cloud resources."},
        _ok({"items": ["b-1"]}),
        _call("get_bucket_acl", bucket_name="b-1"),
    )
    violations = check_recovery_invariant(messages, "D1")
    assert violations
    assert "no preceding error" in violations[0]


def test_error_that_does_not_name_the_fix_fails():
    messages = _row(
        {"role": "system", "content": "Tools: get_bucket_acl(bucket)"},
        {"role": "user", "content": "Find all vulnerable cloud resources."},
        _call("get_bucket_acl", bucket="b-1"),
        _error("schema_violation: unexpected argument(s): bucket"),
        _call("get_bucket_acl", bucket_name="b-1"),
    )
    violations = check_recovery_invariant(messages, "D1")
    assert violations
    assert "does not name it" in violations[0]


def test_clean_trace_must_not_use_a_drifted_name():
    messages = _row(
        {"role": "system", "content": "Tools: get_bucket_acl(bucket)"},
        {"role": "user", "content": "Find all vulnerable cloud resources."},
        _call("get_bucket_acl", bucket_name="b-1"),
    )
    assert clean_violations(messages)


def test_the_old_sft_chosen_file_still_fails_the_invariant():
    """A regression test for the diagnosis itself, not only for the fix.

    If `sft_chosen.jsonl` ever passes this, either it was regenerated with
    recovery turns (good, update the test) or the invariant checker broke.
    """
    if not SFT_CHOSEN.is_file():
        pytest.skip("sft_chosen.jsonl not present")

    rows = [
        json.loads(line)
        for line in SFT_CHOSEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total = sum(
        len(check_recovery_invariant(row["messages"], "D1")) for row in rows
    )
    assert total > 0, (
        "the old SFT set has no violations; the diagnosis in the article "
        "would no longer hold"
    )


@pytest.fixture(scope="module")
def recovery_rows() -> list[dict]:
    if not RECOVERY.is_file():
        pytest.skip("recovery_sft.jsonl not generated; run python -m heal.recovery")
    return [
        json.loads(line)
        for line in RECOVERY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_generated_recovery_data_respects_the_invariant(recovery_rows):
    drift_rows = [row for row in recovery_rows if row["drift"]]
    assert drift_rows, "no drift rows in the recovery set"

    for row in drift_rows:
        assert check_recovery_invariant(row["messages"], row["drift"]) == []
        assert any(
            message.get("role") == "tool"
            and "error" in str(message.get("content", "")).lower()
            for message in row["messages"]
        ), f"seed {row['seed']} has no error observation"


def test_generated_clean_data_contains_no_drifted_names(recovery_rows):
    clean = [row for row in recovery_rows if not row["drift"]]
    assert clean, "no clean rows in the recovery set"

    for row in clean:
        assert clean_violations(row["messages"]) == []


def test_generated_data_is_disjoint_from_the_gate_block(recovery_rows):
    """The gate grades 40000-40059; no training row may use those seeds."""
    reserved = set(range(40000, 40160))
    seeds = {row["seed"] for row in recovery_rows}
    assert not (seeds & reserved)
