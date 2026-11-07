from world.tools import CloudEnv
from heal.drifts.d3_new_required_region import apply_d3, TOOL_SCHEMAS_D3


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
        "security_groups": [{"id": "sg-1", "rules": []}],
    })


def test_d3_requires_region_for_list_buckets():
    drift = apply_d3(make_env())
    result = drift.list_buckets()
    assert "schema_violation" in result["error"]


def test_d3_list_buckets_works_with_region():
    drift = apply_d3(make_env())
    result = drift.list_buckets(region="us-east-1")
    assert result["items"] == ["bucket-1"]


def test_d3_requires_region_for_security_groups():
    drift = apply_d3(make_env())
    result = drift.describe_security_groups()
    assert "schema_violation" in result["error"]


def test_d3_security_groups_work_with_region():
    drift = apply_d3(make_env())
    result = drift.describe_security_groups(region="us-east-1")
    assert len(result["security_groups"]) == 1


def test_d3_schema_documents_required_region():
    schemas = {item["name"]: item["description"] for item in TOOL_SCHEMAS_D3}
    assert "region (str, required)" in schemas["list_buckets"]
    assert "region (str, required)" in schemas["describe_security_groups"]


def test_d3_can_be_disabled():
    env = make_env()
    assert apply_d3(env, enabled=False) is env
