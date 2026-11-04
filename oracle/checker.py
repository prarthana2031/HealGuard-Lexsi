def check_vulnerabilities(snapshot: dict) -> list[str]:
    """Returns the sorted list of vulnerable resource IDs."""
    vulnerable = []
    
    # 1. Check Buckets (Public Access Block overrides ACL/Policy. Deny overrides Allow.)
    for b in snapshot["buckets"]:
        if b["public_access_block"]:
            continue # Safe, regardless of ACL/Policy
            
        if b["policy_deny"]:
            continue # Explicit deny makes it safe
            
        if b["acl_public"] or b["policy_public"]:
            vulnerable.append(b["id"])
            
    # 2. Check IAM Roles (Wildcard Action + Wildcard Resource)
    for r in snapshot["roles"]:
        for p in r["policies"]:
            if p["effect"] == "Allow" and p["action"] == "*" and p["resource"] == "*":
                vulnerable.append(r["id"])
                break # Move to next role
                
    # 3. Check Security Groups (Open to 0.0.0.0/0 on risky ports)
    for sg in snapshot["security_groups"]:
        for rule in sg["ingress"]:
            if rule["cidr"] == "0.0.0.0/0" and rule["port"] in [22, 3389, 3306]:
                vulnerable.append(sg["id"])
                break # Move to next SG
                
    return sorted(list(set(vulnerable)))