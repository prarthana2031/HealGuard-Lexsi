import pytest
import json
import random
from world.snapshot import generate_snapshot
from expert.scripted import run_expert
from oracle.checker import check_vulnerabilities
from world.tools import CloudEnv

def test_determinism():
    snap1 = json.dumps(generate_snapshot(42), sort_keys=True)
    snap2 = json.dumps(generate_snapshot(42), sort_keys=True)
    assert snap1 == snap2

def test_expert_100_percent():
    for seed in range(100):
        snap = generate_snapshot(seed)
        trace = run_expert(snap)
        assert trace["is_correct"] == True
        assert trace["final_answer"] == check_vulnerabilities(snap)

def test_random_policy_floor():
    """A random policy should score <5%."""
    random.seed(42) 
    
    correct = 0
    for seed in range(100):
        snap = generate_snapshot(seed)
        truth = check_vulnerabilities(snap)
        
        all_resources = [b["id"] for b in snap["buckets"]] + [r["id"] for r in snap["roles"]] + [sg["id"] for sg in snap["security_groups"]]
        
        # --- THE FIX: Random policy must guess AT LEAST 1 resource ---
        guess = random.sample(all_resources, k=random.randint(1, len(all_resources)))
        
        if sorted(guess) == truth:
            correct += 1
            
    assert (correct / 100) < 0.05, f"Random policy scored {correct}% (Too easy!)"

def test_none_rate():
    empty_count = 0
    for seed in range(100):
        snap = generate_snapshot(seed)
        if len(check_vulnerabilities(snap)) == 0:
            empty_count += 1
    assert 40 <= empty_count <= 60, f"None-rate is {empty_count}%!"

def test_pagination():
    snap = generate_snapshot(1)
    env = CloudEnv(snap)
    
    res1 = env.list_buckets(limit=1)
    assert len(res1["items"]) == 1
    assert res1["next_token"] is not None
    
    res2 = env.list_buckets(limit=1, next_token=res1["next_token"])
    assert len(res2["items"]) == 1
    assert res2["items"][0] != res1["items"][0]

def test_structured_errors():
    snap = generate_snapshot(1)
    env = CloudEnv(snap)

    res = env.list_buckets(limit="five")
    assert "error" in res
    assert "schema_violation" in res["error"]


def test_calls_after_finish_are_ignored_not_environment_errors():
    """The plan says ignored and flagged — not `env_error`.

    `env_error` is reserved for injected 5xx/timeout. Labelling a post-finish
    call as one used to send it to the wrong place in the training pool.
    """
    env = CloudEnv(generate_snapshot(1))
    env.finish([])

    res = env.list_buckets()
    assert res.get("ignored") is True
    assert "error" not in res
    assert env.ignored_calls == 1

    # A second finish must not overwrite the first answer.
    again = env.finish(["something-else"])
    assert again.get("ignored") is True
    assert env.ignored_calls == 2


def test_tool_output_is_capped():
    snapshot = {
        "seed": 0,
        "buckets": [],
        "roles": [],
        "security_groups": [
            {"id": f"sg-0-{i}", "ingress": [{"port": 443, "cidr": "10.0.0.0/8"}]}
            for i in range(40)
        ],
    }
    res = CloudEnv(snapshot).describe_security_groups()

    assert len(res["security_groups"]) == 25
    assert res["truncated_items"] == 15
