"""D5 fail-closed bucket tests."""

from heal.drifts.d5_silent_empty import D5SilentEmptyEnv, apply_d5
from heal.drifts.registry import apply_drift, get_tool_schemas
from world.tools import CloudEnv


def make_env():
    return CloudEnv(
        {
            "buckets": [
                {
                    "id": "bucket-1",
                    "acl_public": True,
                    "policy_public": False,
                    "policy_deny": False,
                    "public_access_block": False,
                }
            ],
            "roles": [],
            "security_groups": [],
        }
    )


def test_stale_argument_returns_an_empty_payload_not_an_error():
    drift = apply_d5(make_env())

    result = drift.get_bucket_acl(bucket="bucket-1")

    assert result == {}
    assert "error" not in result


def test_new_argument_returns_real_data():
    drift = apply_d5(make_env())

    assert drift.get_bucket_acl(bucket_name="bucket-1") == {"acl_public": True}
    assert drift.get_bucket_policy(bucket_name="bucket-1") == {
        "policy_public": False,
        "policy_deny": False,
    }
    assert drift.get_public_access_block(bucket_name="bucket-1") == {
        "public_access_block": False
    }


def test_post_finish_call_is_ignored_not_silently_empty():
    base = make_env()
    drift = apply_d5(base)
    drift.finish([])

    result = drift.get_bucket_acl(bucket="bucket-1")

    assert result.get("ignored") is True
    assert base.ignored_calls == 1


def test_registry_maps_d5_to_the_silent_environment():
    runtime = apply_drift(make_env(), "D5")

    assert isinstance(runtime, D5SilentEmptyEnv)


def test_d5_schema_keeps_the_clean_bucket_contract():
    description = next(
        schema["description"]
        for schema in get_tool_schemas("D5")
        if schema["name"] == "get_bucket_acl"
    )

    assert "bucket (str)" in description
    assert "bucket_name" not in description


def test_d5_failure_is_residual_not_schema_violation():
    """The empty payload is the failure, not an invalid argument."""
    from eval.run_agent import run_episode
    from oracle.checker import check_vulnerabilities
    from world.snapshot import generate_snapshot

    seed = next(
        seed
        for seed in range(50)
        if check_vulnerabilities(generate_snapshot(seed))
    )
    record = run_episode(seed, "D5", "stale", "dev")

    assert record["failure_mode"] != "schema_violation"


def test_apply_d5_can_be_disabled():
    env = make_env()

    assert apply_d5(env, enabled=False) is env
