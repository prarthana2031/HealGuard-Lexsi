import pytest

from world.tools import CloudEnv
from heal.drifts.registry import (
    apply_drift,
    build_runtime,
    get_tool_schemas,
)


def make_env():
    return CloudEnv({
        "buckets": [{
            "id": "bucket-1",
            "acl_public": True,
            "policy_public": False,
            "policy_deny": False,
            "public_access_block": {},
        }],
        "roles": [],
        "security_groups": [],
    })


@pytest.mark.parametrize("drift", ["D1", "D2", "D3", "D4"])
def test_registry_returns_matching_schema_for_each_drift(drift):
    env, schemas = build_runtime(make_env(), drift)
    names = [schema["name"] for schema in schemas]

    assert env is not None
    assert names

    if drift == "D1":
        description = next(
            s["description"] for s in schemas
            if s["name"] == "get_bucket_acl"
        )
        assert "bucket_name" in description

    elif drift == "D2":
        description = next(
            s["description"] for s in schemas
            if s["name"] == "list_buckets"
        )
        assert "bucket_list" in description
        assert "continuation" in description

    elif drift == "D3":
        description = next(
            s["description"] for s in schemas
            if s["name"] == "list_buckets"
        )
        assert "region" in description

    elif drift == "D4":
        assert "get_acl" in names
        assert "get_bucket_acl" not in names


def test_clean_runtime_returns_original_environment_and_schema():
    env = make_env()
    runtime_env, schemas = build_runtime(env, "clean")

    assert runtime_env is env
    assert schemas == get_tool_schemas()


def test_disabled_drift_returns_clean_runtime():
    env = make_env()
    runtime_env, schemas = build_runtime(env, "D2", enabled=False)

    assert runtime_env is env
    assert schemas == get_tool_schemas()


def test_unknown_drift_is_rejected():
    with pytest.raises(ValueError, match="Unknown drift"):
        apply_drift(make_env(), "D5")


def test_schema_results_are_copies():
    schemas = get_tool_schemas("D1")
    schemas[0]["description"] = "changed"

    fresh = get_tool_schemas("D1")
    assert fresh[0]["description"] != "changed"
