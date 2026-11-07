import json
from world.tools import CloudEnv, TOOL_SCHEMAS
from oracle.checker import check_vulnerabilities
from heal.drifts.dispatcher import create_dispatcher

def run_expert(snapshot: dict, inject_error: bool = False, drift: str = None) -> dict:
    """
    Runs the scripted expert. If drift is provided (e.g., 'D1'), 
    it wraps the env and updates the tool calls to match the new API.
    """
    base_env = CloudEnv(snapshot)
    
    if drift:
        dispatcher = create_dispatcher(base_env, drift=drift)
        env = dispatcher.env
        # Use drift-specific schemas for the prompt
        tool_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in dispatcher.schemas])
    else:
        env = base_env
        tool_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in TOOL_SCHEMAS])

    ground_truth = check_vulnerabilities(snapshot)
    messages = []
    
    sys_prompt = f"You are a cloud auditor. Output exactly like this: <tool_call>\n{{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n</tool_call>\nAvailable Tools:\n{tool_desc}"
    messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": "Find all vulnerable cloud resources."})

    # Determine argument names based on drift
    bucket_arg = "bucket_name" if drift in ["D1", "D4"] else "bucket"
    region_arg = ', "region": "us-east-1"' if drift == "D3" else ""
    acl_tool = "get_acl" if drift == "D4" else "get_bucket_acl"

    # 1. List buckets
    call_str = f"<tool_call>\n{{\"name\": \"list_buckets\", \"arguments\": {{{region_arg}}}}}\n</tool_call>"
    messages.append({"role": "assistant", "content": call_str})
    
    if inject_error:
        messages.append({"role": "tool", "content": "Error: 503 Service Unavailable"})
        messages.append({"role": "assistant", "content": call_str})
    
    # Execute tool via dispatcher or env
    kwargs = {"region": "us-east-1"} if drift == "D3" else {}
    res = env.list_buckets(**kwargs)
    messages.append({"role": "tool", "content": json.dumps(res)})

    # 2. Pagination & Resource Checking
    buckets = res.get("items", res.get("bucket_list", []))
    while res.get("next_token") or res.get("continuation"):
        next_tok = res.get("next_token", res.get("continuation"))
        call_str = f"<tool_call>\n{{\"name\": \"list_buckets\", \"arguments\": {{\"next_token\": \"{next_tok}\"{region_arg}}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call_str})
        res = env.list_buckets(next_token=next_tok, **kwargs)
        messages.append({"role": "tool", "content": json.dumps(res)})
        buckets.extend(res.get("items", res.get("bucket_list", [])))

    for b_id in buckets:
        # Check ACL
        call = f"<tool_call>\n{{\"name\": \"{acl_tool}\", \"arguments\": {{\"{bucket_arg}\": \"{b_id}\"}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call})
        messages.append({"role": "tool", "content": json.dumps(getattr(env, acl_tool)(**{bucket_arg: b_id}))})
        
        # Check Policy
        call = f"<tool_call>\n{{\"name\": \"get_bucket_policy\", \"arguments\": {{\"{bucket_arg}\": \"{b_id}\"}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call})
        messages.append({"role": "tool", "content": json.dumps(env.get_bucket_policy(**{bucket_arg: b_id}))})
        
        # Check Block
        call = f"<tool_call>\n{{\"name\": \"get_public_access_block\", \"arguments\": {{\"{bucket_arg}\": \"{b_id}\"}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call})
        messages.append({"role": "tool", "content": json.dumps(env.get_public_access_block(**{bucket_arg: b_id}))})

    # 3. IAM Roles
    call = f"<tool_call>\n{{\"name\": \"list_iam_roles\", \"arguments\": {{}}}}\n</tool_call>"
    messages.append({"role": "assistant", "content": call})
    res = env.list_iam_roles()
    messages.append({"role": "tool", "content": json.dumps(res)})
    
    for r_id in res.get("roles", []):
        call = f"<tool_call>\n{{\"name\": \"get_role_policies\", \"arguments\": {{\"role_name\": \"{r_id}\"}}}}\n</tool_call>"
        messages.append({"role": "assistant", "content": call})
        messages.append({"role": "tool", "content": json.dumps(env.get_role_policies(r_id))})

    # 4. Security Groups
    call = f"<tool_call>\n{{\"name\": \"describe_security_groups\", \"arguments\": {{{region_arg}}}}}\n</tool_call>"
    messages.append({"role": "assistant", "content": call})
    kwargs_sg = {"region": "us-east-1"} if drift == "D3" else {}
    res = env.describe_security_groups(**kwargs_sg)
    messages.append({"role": "tool", "content": json.dumps(res)})

    # 5. Finish
    finish_call = f"<tool_call>\n{{\"name\": \"finish\", \"arguments\": {{\"answer\": {json.dumps(ground_truth)}}}}}\n</tool_call>"
    messages.append({"role": "assistant", "content": finish_call})
    env.finish(ground_truth)

    return {
        "task_id": f"task_{snapshot['seed']}",
        "seed": snapshot["seed"],
        "split": "train_clean" if not drift else "drift_train",
        "drift": drift,
        "messages": messages,
        "final_answer": ground_truth,
        "is_correct": True
    }