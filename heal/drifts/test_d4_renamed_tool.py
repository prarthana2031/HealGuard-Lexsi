from world.tools import CloudEnv
from heal.drifts.d4_renamed_tool import apply_d4, TOOL_SCHEMAS_D4


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


def test_d4_new_tool_name_works():
    drift = apply_d4(make_env())
    assert drift.get_acl("bucket-1") == {"acl_public": True}


def test_d4_old_tool_name_returns_unknown_tool_error():
    drift = apply_d4(make_env())
    result = drift.get_bucket_acl(bucket="bucket-1")
    assert "unknown_tool" in result["error"]
    assert "get_bucket_acl" in result["error"]
    assert "get_acl" in result["error"]


def test_d4_schema_hides_old_name_and_exposes_new_name():
    names = [item["name"] for item in TOOL_SCHEMAS_D4]
    assert "get_acl" in names
    assert "get_bucket_acl" not in names


def test_d4_can_be_disabled():
    env = make_env()
    assert apply_d4(env, enabled=False) is env
