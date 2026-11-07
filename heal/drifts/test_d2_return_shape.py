from world.tools import CloudEnv
from heal.drifts.d2_return_shape import apply_d2, TOOL_SCHEMAS_D2


def make_env():
    return CloudEnv({
        "buckets": [
            {"id": "bucket-1"},
            {"id": "bucket-2"},
            {"id": "bucket-3"},
        ],
        "roles": [],
        "security_groups": [],
    })


def test_d2_renames_response_keys():
    drift = apply_d2(make_env())

    result = drift.list_buckets(limit=2)

    assert result == {
        "bucket_list": ["bucket-1", "bucket-2"],
        "continuation": "2",
    }
    assert "items" not in result
    assert "next_token" not in result


def test_d2_pagination_still_works_with_new_keys():
    drift = apply_d2(make_env())

    page1 = drift.list_buckets(limit=2)
    page2 = drift.list_buckets(
        limit=2,
        next_token=page1["continuation"],
    )

    assert page1["bucket_list"] == ["bucket-1", "bucket-2"]
    assert page2["bucket_list"] == ["bucket-3"]
    assert page2["continuation"] is None


def test_d2_preserves_underlying_schema_errors():
    drift = apply_d2(make_env())

    result = drift.list_buckets(limit="invalid")

    assert "error" in result
    assert "schema_violation" in result["error"]


def test_d2_preserves_finished_environment_errors():
    env = make_env()
    drift = apply_d2(env)
    drift.finish([])

    result = drift.list_buckets()

    assert "error" in result
    assert "env_error" in result["error"]


def test_d2_updates_model_tool_description():
    schemas = {
        item["name"]: item["description"]
        for item in TOOL_SCHEMAS_D2
    }

    description = schemas["list_buckets"]
    assert "bucket_list" in description
    assert "continuation" in description


def test_d2_can_be_disabled():
    env = make_env()
    assert apply_d2(env, enabled=False) is env
