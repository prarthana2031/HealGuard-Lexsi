from world.tools import CloudEnv
from heal.drifts.dispatcher import create_dispatcher


def make_env():
    return CloudEnv({
        "buckets": [
            {
                "id": "bucket-1",
                "acl_public": True,
                "policy_public": False,
                "policy_deny": False,
                "public_access_block": {},
            },
            {
                "id": "bucket-2",
                "acl_public": False,
                "policy_public": False,
                "policy_deny": False,
                "public_access_block": {},
            },
        ],
        "roles": [],
        "security_groups": [],
    })


def test_clean_dispatcher_executes_real_tool():
    dispatcher = create_dispatcher(make_env())
    result = dispatcher.dispatch({
        "name": "get_bucket_acl",
        "arguments": {"bucket": "bucket-1"},
    })
    assert result == {"acl_public": True}


def test_d1_dispatcher_uses_renamed_argument():
    dispatcher = create_dispatcher(make_env(), "D1")

    result = dispatcher.dispatch({
        "name": "get_bucket_acl",
        "arguments": {"bucket_name": "bucket-1"},
    })
    assert result == {"acl_public": True}

    old_call = dispatcher.dispatch({
        "name": "get_bucket_acl",
        "arguments": {"bucket": "bucket-1"},
    })
    assert "schema_violation" in old_call["error"]


def test_d2_dispatcher_uses_changed_return_shape():
    dispatcher = create_dispatcher(make_env(), "D2")

    result = dispatcher.dispatch({
        "name": "list_buckets",
        "arguments": {"limit": 1},
    })
    assert result == {
        "bucket_list": ["bucket-1"],
        "continuation": "1",
    }


def test_d3_dispatcher_requires_region():
    dispatcher = create_dispatcher(make_env(), "D3")

    result = dispatcher.dispatch({
        "name": "list_buckets",
        "arguments": {},
    })
    assert "schema_violation" in result["error"]

    valid = dispatcher.dispatch({
        "name": "list_buckets",
        "arguments": {"region": "us-east-1"},
    })
    assert valid["items"] == ["bucket-1", "bucket-2"]


def test_d4_dispatcher_uses_new_tool_name():
    dispatcher = create_dispatcher(make_env(), "D4")

    result = dispatcher.dispatch({
        "name": "get_acl",
        "arguments": {"bucket": "bucket-1"},
    })
    assert result == {"acl_public": True}

    old = dispatcher.dispatch({
        "name": "get_bucket_acl",
        "arguments": {"bucket": "bucket-1"},
    })
    assert "unknown_tool" in old["error"]
    assert "get_acl" in old["error"]


def test_dispatcher_rejects_malformed_calls():
    dispatcher = create_dispatcher(make_env())

    assert "schema_violation" in dispatcher.dispatch(None)["error"]
    assert "schema_violation" in dispatcher.dispatch({
        "name": "list_buckets",
        "arguments": [],
    })["error"]


def test_dispatcher_rejects_unknown_tool():
    dispatcher = create_dispatcher(make_env())
    result = dispatcher.dispatch({
        "name": "delete_everything",
        "arguments": {},
    })
    assert "unknown_tool" in result["error"]


def test_dispatcher_can_disable_drift():
    env = make_env()
    dispatcher = create_dispatcher(env, "D1", enabled=False)

    result = dispatcher.dispatch({
        "name": "get_bucket_acl",
        "arguments": {"bucket": "bucket-1"},
    })
    assert result == {"acl_public": True}
