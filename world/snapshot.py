import random

def generate_snapshot(seed: int) -> dict:
    """Generates a deterministic fake cloud account for a given seed."""
    rng = random.Random(seed)
    
    # 50% chance of having NO vulnerabilities (The "none" rate)
    is_clean = rng.random() < 0.5
    
    snapshot = {
        "seed": seed,
        "buckets": [],
        "roles": [],
        "security_groups": []
    }
    
    # --- Generate F1: Buckets (Now 3) ---
    for i in range(3):
        bucket = {
            "id": f"bucket-{seed}-{i}",
            "acl_public": False,
            "policy_public": False,
            "policy_deny": False,
            "public_access_block": False
        }
        
        # If we want a vulnerable task, make one bucket public
        if not is_clean and i == 0:
            bucket["acl_public"] = True 
            
        # Add some noise: public ACL but overridden by Access Block
        if i == 1 and rng.random() < 0.5:
            bucket["acl_public"] = True
            bucket["public_access_block"] = True 
            
        # Add explicit deny noise
        if i == 2 and rng.random() < 0.5:
            bucket["policy_public"] = True
            bucket["policy_deny"] = True 
            
        snapshot["buckets"].append(bucket)
        
    # --- Generate F2: IAM Roles (Now 3) ---
    for i in range(3):
        role = {
            "id": f"role-{seed}-{i}",
            "policies": []
        }
        if not is_clean and i == 0:
            role["policies"].append({"action": "*", "resource": "*", "effect": "Allow"})
        else:
            role["policies"].append({"action": "s3:GetObject", "resource": "arn:aws:s3:::safe", "effect": "Allow"})
        snapshot["roles"].append(role)

    # --- Generate F4: Security Groups (Now 3) ---
    for i in range(3):
        sg = {
            "id": f"sg-{seed}-{i}",
            "ingress": []
        }
        if not is_clean and i == 0:
            sg["ingress"].append({"port": 22, "cidr": "0.0.0.0/0"}) 
        else:
            sg["ingress"].append({"port": 443, "cidr": "0.0.0.0/0"}) 
        snapshot["security_groups"].append(sg)
        
    return snapshot