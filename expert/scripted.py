import json
from world.tools import CloudEnv, TOOL_SCHEMAS
from oracle.checker import check_vulnerabilities

def run_expert(snapshot: dict, inject_error: bool = False) -> dict:
    env = CloudEnv(snapshot)
    ground_truth = check_vulnerabilities(snapshot)
    messages = []
    
    # 1. System Prompt (Matches Person A's S2 test)
    tool_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in TOOL_SCHEMAS])
    sys_prompt = f"You are a cloud auditor. Output exactly like this: <tool_call>\n{{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n</tool_call>\nAvailable Tools:\n{tool_desc}"
    messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": "Find all vulnerable cloud resources."})

    # 2. Expert logic: List buckets (with optional Error Recovery)
    call_str = "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"
    messages.append({"role": "assistant", "content": call_str})
    
    if inject_error:
        # Simulate a 503 Error, then retry
        messages.append({"role": "tool", "content": "Error: 503 Service Unavailable"})
        messages.append({"role": "assistant", "content": call_str})
    
    # Actual tool execution
    res = env.list_buckets()
    messages.append({"role": "tool", "content": json.dumps(res)})

    # 3. Expert logic: Check each bucket (Pagination aware)
    buckets = res.get("items", [])
    while res.get("next_token"):
        call_str = f"<tool_call>\n{{\"name\": \"list_buckets\", \"arguments\": {{\"next_token\": \"{res['next_token']}\"}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call_str})
        res = env.list_buckets(next_token=res["next_token"])
        messages.append({"role": "tool", "content": json.dumps(res)})
        buckets.extend(res.get("items", []))

    for b_id in buckets:
        # Check ACL
        call = f"<tool_call>\n{{\"name\": \"get_bucket_acl\", \"arguments\": {{\"bucket\": \"{b_id}\"}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call})
        messages.append({"role": "tool", "content": json.dumps(env.get_bucket_acl(b_id))})
        
        # Check Policy
        call = f"<tool_call>\n{{\"name\": \"get_bucket_policy\", \"arguments\": {{\"bucket\": \"{b_id}\"}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call})
        messages.append({"role": "tool", "content": json.dumps(env.get_bucket_policy(b_id))})
        
        # Check Block
        call = f"<tool_call>\n{{\"name\": \"get_public_access_block\", \"arguments\": {{\"bucket\": \"{b_id}\"}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call})
        messages.append({"role": "tool", "content": json.dumps(env.get_public_access_block(b_id))})

    # 4. Expert logic: Check IAM roles
    call = "<tool_call>\n{\"name\": \"list_iam_roles\", \"arguments\": {}}\n</tool_call>"
    messages.append({"role": "assistant", "content": call})
    res = env.list_iam_roles()
    messages.append({"role": "tool", "content": json.dumps(res)})
    
    for r_id in res["roles"]:
        call = f"<tool_call>\n{{\"name\": \"get_role_policies\", \"arguments\": {{\"role_name\": \"{r_id}\"}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call})
        messages.append({"role": "tool", "content": json.dumps(env.get_role_policies(r_id))})

    # 5. Expert logic: Check Security Groups
    call = "<tool_call>\n{\"name\": \"describe_security_groups\", \"arguments\": {}}\n</tool_call>"
    messages.append({"role": "assistant", "content": call})
    res = env.describe_security_groups()
    messages.append({"role": "tool", "content": json.dumps(res)})

    # 6. Finish
    finish_call = f"<tool_call>\n{{\"name\": \"finish\", \"arguments\": {{\"answer\": {json.dumps(ground_truth)}}}}}\n</tool_call>"
    messages.append({"role": "assistant", "content": finish_call})
    env.finish(ground_truth)

    return {
        "task_id": f"task_{snapshot['seed']}",
        "seed": snapshot["seed"],
        "split": "train_clean",
        "messages": messages,
        "final_answer": ground_truth,
        "is_correct": True
    }