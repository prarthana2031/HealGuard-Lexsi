from world.tools import CloudEnv
from heal.drifts.d1_renamed_arg import apply_d1, TOOL_SCHEMAS_D1


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


def test_d1_renames_all_three_bucket_arguments():
    env = make_env()
    drift = apply_d1(env)

    assert drift.get_bucket_acl("bucket-1") == {"acl_public": True}
    assert drift.get_bucket_policy("bucket-1") == {
        "policy_public": False,
        "policy_deny": False,
    }
    assert drift.get_public_access_block("bucket-1") == {
        "public_access_block": {}
    }


def test_old_bucket_argument_is_rejected():
    drift = apply_d1(make_env())

    result = drift.get_bucket_acl(bucket="bucket-1")

    assert result["error"] == "schema_violation: invalid arguments"
    assert "bucket_name" not in result["error"]


def test_d1_keeps_the_same_snapshot_and_finished_state():
    env = make_env()
    drift = apply_d1(env)

    assert drift.snapshot is env.snapshot
    drift.finish([])
    assert env.finished is True


def test_d1_can_be_disabled():
    env = make_env()
    assert apply_d1(env, enabled=False) is env


def test_d1_tool_schemas_show_bucket_name():
    changed = {
        item["name"]: item["description"]
        for item in TOOL_SCHEMAS_D1
    }

    for name in (
        "get_bucket_acl",
        "get_bucket_policy",
        "get_public_access_block",
    ):
        assert "bucket_name (str)" in changed[name]
        assert "bucket (str)" not in changed[name]
