import pytest
from oracle.checker import check_vulnerabilities

# 30 Hand-Computed Test Cases
# Each tuple is: (Test Name, Snapshot Dictionary, Expected List of Vulnerable IDs)
HAND_CASES = [
    # --- BUCKETS (F1) ---
    ("01_empty_account", {"buckets": [], "roles": [], "security_groups": []}, []),
    ("02_safe_bucket", {"buckets": [{"id": "b1", "acl_public": False, "policy_public": False, "policy_deny": False, "public_access_block": False}], "roles": [], "security_groups": []}, []),
    ("03_public_acl", {"buckets": [{"id": "b1", "acl_public": True, "policy_public": False, "policy_deny": False, "public_access_block": False}], "roles": [], "security_groups": []}, ["b1"]),
    ("04_public_policy", {"buckets": [{"id": "b1", "acl_public": False, "policy_public": True, "policy_deny": False, "public_access_block": False}], "roles": [], "security_groups": []}, ["b1"]),
    ("05_both_public", {"buckets": [{"id": "b1", "acl_public": True, "policy_public": True, "policy_deny": False, "public_access_block": False}], "roles": [], "security_groups": []}, ["b1"]),
    ("06_block_overrides_acl", {"buckets": [{"id": "b1", "acl_public": True, "policy_public": False, "policy_deny": False, "public_access_block": True}], "roles": [], "security_groups": []}, []),
    ("07_block_overrides_policy", {"buckets": [{"id": "b1", "acl_public": False, "policy_public": True, "policy_deny": False, "public_access_block": True}], "roles": [], "security_groups": []}, []),
    ("08_explicit_deny_overrides_acl", {"buckets": [{"id": "b1", "acl_public": True, "policy_public": False, "policy_deny": True, "public_access_block": False}], "roles": [], "security_groups": []}, []),
    ("09_explicit_deny_overrides_policy", {"buckets": [{"id": "b1", "acl_public": False, "policy_public": True, "policy_deny": True, "public_access_block": False}], "roles": [], "security_groups": []}, []),
    ("10_multiple_buckets", {"buckets": [
        {"id": "b1", "acl_public": False, "policy_public": False, "policy_deny": False, "public_access_block": False},
        {"id": "b2", "acl_public": True, "policy_public": False, "policy_deny": False, "public_access_block": False}
    ], "roles": [], "security_groups": []}, ["b2"]),
    ("11_no_such_bucket_policy_edge", {"buckets": [{"id": "b1", "acl_public": False, "policy_public": False, "policy_deny": False, "public_access_block": False}], "roles": [], "security_groups": []}, []), # Safely handles missing policy

    # --- IAM ROLES (F2) ---
    ("12_safe_role", {"buckets": [], "roles": [{"id": "r1", "policies": [{"action": "s3:GetObject", "resource": "arn:aws:s3:::b1", "effect": "Allow"}]}], "security_groups": []}, []),
    ("13_wildcard_role", {"buckets": [], "roles": [{"id": "r1", "policies": [{"action": "*", "resource": "*", "effect": "Allow"}]}], "security_groups": []}, ["r1"]),
    ("14_wildcard_action_only", {"buckets": [], "roles": [{"id": "r1", "policies": [{"action": "*", "resource": "arn:aws:s3:::b1", "effect": "Allow"}]}], "security_groups": []}, []),
    ("15_wildcard_resource_only", {"buckets": [], "roles": [{"id": "r1", "policies": [{"action": "s3:GetObject", "resource": "*", "effect": "Allow"}]}], "security_groups": []}, []),
    ("16_multi_policy_safe", {"buckets": [], "roles": [{"id": "r1", "policies": [{"action": "s3:Get", "resource": "b1", "effect": "Allow"}, {"action": "s3:Put", "resource": "b1", "effect": "Allow"}]}], "security_groups": []}, []),
    ("17_multi_policy_one_bad", {"buckets": [], "roles": [{"id": "r1", "policies": [{"action": "s3:Get", "resource": "b1", "effect": "Allow"}, {"action": "*", "resource": "*", "effect": "Allow"}]}], "security_groups": []}, ["r1"]),
    ("18_wildcard_deny", {"buckets": [], "roles": [{"id": "r1", "policies": [{"action": "*", "resource": "*", "effect": "Deny"}]}], "security_groups": []}, []),
    ("19_two_bad_roles", {"buckets": [], "roles": [
        {"id": "r1", "policies": [{"action": "*", "resource": "*", "effect": "Allow"}]},
        {"id": "r2", "policies": [{"action": "*", "resource": "*", "effect": "Allow"}]}
    ], "security_groups": []}, ["r1", "r2"]),
    ("20_empty_role", {"buckets": [], "roles": [{"id": "r1", "policies": []}], "security_groups": []}, []),

    # --- SECURITY GROUPS (F4) ---
    ("21_safe_sg", {"buckets": [], "roles": [], "security_groups": [{"id": "sg1", "ingress": [{"port": 443, "cidr": "0.0.0.0/0"}]}]}, []),
    ("22_open_ssh", {"buckets": [], "roles": [], "security_groups": [{"id": "sg1", "ingress": [{"port": 22, "cidr": "0.0.0.0/0"}]}]}, ["sg1"]),
    ("23_open_rdp", {"buckets": [], "roles": [], "security_groups": [{"id": "sg1", "ingress": [{"port": 3389, "cidr": "0.0.0.0/0"}]}]}, ["sg1"]),
    ("24_open_mysql", {"buckets": [], "roles": [], "security_groups": [{"id": "sg1", "ingress": [{"port": 3306, "cidr": "0.0.0.0/0"}]}]}, ["sg1"]),
    ("25_open_http_safe", {"buckets": [], "roles": [], "security_groups": [{"id": "sg1", "ingress": [{"port": 80, "cidr": "0.0.0.0/0"}]}]}, []), # Port 80 is not in [22, 3389, 3306]
    ("26_ssh_restricted_ip", {"buckets": [], "roles": [], "security_groups": [{"id": "sg1", "ingress": [{"port": 22, "cidr": "10.0.0.0/8"}]}]}, []), # CIDR is not 0.0.0.0/0
    ("27_multi_rule_sg", {"buckets": [], "roles": [], "security_groups": [{"id": "sg1", "ingress": [{"port": 443, "cidr": "0.0.0.0/0"}, {"port": 22, "cidr": "0.0.0.0/0"}]}]}, ["sg1"]),
    ("28_empty_page", {"buckets": [], "roles": [], "security_groups": []}, []), # PDF says "one empty page"
    
    # --- COMBINATIONS ---
    ("29_all_vulnerable", {
        "buckets": [{"id": "b1", "acl_public": True, "policy_public": False, "policy_deny": False, "public_access_block": False}],
        "roles": [{"id": "r1", "policies": [{"action": "*", "resource": "*", "effect": "Allow"}]}],
        "security_groups": [{"id": "sg1", "ingress": [{"port": 22, "cidr": "0.0.0.0/0"}]}]
    }, ["b1", "r1", "sg1"]),
    ("30_all_safe", {
        "buckets": [{"id": "b1", "acl_public": False, "policy_public": False, "policy_deny": False, "public_access_block": False}],
        "roles": [{"id": "r1", "policies": [{"action": "s3:Get", "resource": "b1", "effect": "Allow"}]}],
        "security_groups": [{"id": "sg1", "ingress": [{"port": 443, "cidr": "0.0.0.0/0"}]}]
    }, []),
]

@pytest.mark.parametrize("name, snapshot, expected", HAND_CASES)
def test_hand_oracle(name, snapshot, expected):
    """PDF Requirement: 30 cases computed by hand, checker matches all 30."""
    # Ensure snapshot has 'seed' so it doesn't crash if the oracle expects it
    snapshot["seed"] = 999 
    result = check_vulnerabilities(snapshot)
    assert result == expected, f"Failed case {name}: expected {expected}, got {result}"