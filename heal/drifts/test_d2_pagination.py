"""D2 pagination drift tests.

D2 keeps the clean response envelope (``items`` / ``next_token``) and silently
caps ``list_buckets`` at one bucket per page. A caller that does not follow the
token misses buckets without seeing an error.
"""

import pytest

from heal.drifts.d2_pagination import D2PaginatedEnv, apply_d2
from heal.drifts.dispatcher import create_dispatcher
from heal.drifts.registry import apply_drift, get_tool_schemas
from world.tools import CloudEnv


def make_env(count: int = 3) -> CloudEnv:
    return CloudEnv(
        {
            "buckets": [{"id": f"bucket-{index}"} for index in range(1, count + 1)],
            "roles": [],
            "security_groups": [],
        }
    )


def test_requested_limit_is_forced_to_one():
    drift = apply_d2(make_env())

    page = drift.list_buckets(limit=100)

    assert page["items"] == ["bucket-1"]
    assert page["next_token"] == "1"


def test_first_page_contains_one_bucket():
    drift = apply_d2(make_env())

    page = drift.list_buckets()

    assert len(page["items"]) == 1


def test_valid_next_token_is_returned_when_more_buckets_exist():
    drift = apply_d2(make_env())

    page = drift.list_buckets(limit=100)

    assert page["next_token"] is not None
    assert page["next_token"] == "1"


def test_last_page_has_no_next_token():
    drift = apply_d2(make_env(count=1))

    page = drift.list_buckets(limit=100)

    assert page["items"] == ["bucket-1"]
    assert page["next_token"] is None


def test_next_call_preserves_and_uses_the_token():
    drift = apply_d2(make_env())

    first = drift.list_buckets(limit=100)
    second = drift.list_buckets(limit=100, next_token=first["next_token"])

    assert first["items"] == ["bucket-1"]
    assert second["items"] == ["bucket-2"]
    assert second["next_token"] == "2"


def test_repeated_pagination_retrieves_every_bucket_once():
    drift = apply_d2(make_env(count=4))

    seen = []
    token = None
    while True:
        page = drift.list_buckets(limit=100, next_token=token)
        seen.extend(page["items"])
        token = page["next_token"]
        if token is None:
            break

    assert seen == ["bucket-1", "bucket-2", "bucket-3", "bucket-4"]
    assert len(seen) == len(set(seen))


def test_response_keys_are_the_clean_envelope():
    drift = apply_d2(make_env())

    page = drift.list_buckets(limit=100)

    assert set(page) == {"items", "next_token"}


def test_invalid_limit_keeps_the_structured_schema_error():
    """Forcing the page size must not swallow a malformed call."""
    drift = apply_d2(make_env())

    result = drift.list_buckets(limit="one hundred")

    assert "error" in result
    assert "schema_violation" in result["error"]


def test_post_finish_call_is_ignored_on_the_shared_environment():
    base = make_env()
    drift = apply_d2(base)

    drift.finish([])
    result = drift.list_buckets(limit=100)

    assert base.finished is True
    assert result.get("ignored") is True
    assert base.ignored_calls == 1


def test_apply_d2_can_be_disabled():
    env = make_env()

    assert apply_d2(env, enabled=False) is env


def test_registry_maps_d2_to_the_paginated_environment():
    runtime = apply_drift(make_env(), "D2")

    assert isinstance(runtime, D2PaginatedEnv)


def test_registry_exposes_the_standard_pagination_schema():
    schemas = get_tool_schemas("D2")
    description = next(
        schema["description"] for schema in schemas if schema["name"] == "list_buckets"
    )

    assert "limit" in description
    assert "next_token" in description
    assert "bucket_list" not in description
    assert "continuation" not in description


def test_dispatcher_exposes_the_standard_pagination_schema():
    dispatcher = create_dispatcher(make_env(), "D2")
    description = next(
        schema["description"]
        for schema in dispatcher.schemas
        if schema["name"] == "list_buckets"
    )

    assert "limit" in description
    assert "next_token" in description
    assert "bucket_list" not in description


def test_dispatcher_paginates_with_the_clean_envelope():
    dispatcher = create_dispatcher(make_env(), "D2")

    first = dispatcher.dispatch(
        {"name": "list_buckets", "arguments": {"limit": 100}}
    )
    second = dispatcher.dispatch(
        {
            "name": "list_buckets",
            "arguments": {"limit": 100, "next_token": first["next_token"]},
        }
    )

    assert first == {"items": ["bucket-1"], "next_token": "1"}
    assert second == {"items": ["bucket-2"], "next_token": "2"}


@pytest.mark.parametrize(
    "drift,needle",
    [
        ("D1", "bucket_name"),
        ("D3", "region"),
        ("D4", "get_acl"),
    ],
)
def test_other_drifts_are_unchanged(drift, needle):
    schemas = get_tool_schemas(drift)
    names = [schema["name"] for schema in schemas]
    descriptions = " ".join(schema["description"] for schema in schemas)

    assert needle in descriptions or needle in names
