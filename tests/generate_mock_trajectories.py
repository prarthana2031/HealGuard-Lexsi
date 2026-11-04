import json
import os

os.makedirs("tests/fixtures", exist_ok=True)

trajectories = []
sys_prompt = "You are a helpful cloud auditor with access to tools. Output exactly in this format: <tool_call>\n{\"name\": \"tool_name\", \"arguments\": {}}\n</tool_call>"

# 1 & 2: Wrong Tool
trajectories.append({"task_id": "dev_01_wrong_tool", "seed": 101, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "List buckets."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_iam_roles\", \"arguments\": {}}\n</tool_call>"}], "final_answer": "[]", "is_correct": False})
trajectories.append({"task_id": "dev_02_wrong_tool", "seed": 102, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "List IAM roles."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"}], "final_answer": "[]", "is_correct": False})

# 3 & 4: Schema Violation
trajectories.append({"task_id": "dev_03_schema_violation", "seed": 103, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "List buckets."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {\"limit\": \"five\"}}\n</tool_call>"}], "final_answer": "[]", "is_correct": False})
trajectories.append({"task_id": "dev_04_schema_violation", "seed": 104, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Get bucket policy for bucket-1."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_bucket_policy\", \"arguments\": {}}\n</tool_call>"}], "final_answer": "[]", "is_correct": False})

# 5 & 6: Wrong Order
trajectories.append({"task_id": "dev_05_wrong_order", "seed": 105, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Find public buckets."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_bucket_policy\", \"arguments\": {\"bucket\": \"bucket-1\"}}\n</tool_call>"}], "final_answer": "[]", "is_correct": False})
trajectories.append({"task_id": "dev_06_wrong_order", "seed": 106, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Check role policies."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_role_policies\", \"arguments\": {\"role_name\": \"admin-role\"}}\n</tool_call>"}], "final_answer": "[]", "is_correct": False})

# 7 & 8: Premature Finish
trajectories.append({"task_id": "dev_07_premature_finish", "seed": 107, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "List buckets."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"finish\", \"arguments\": {\"answer\": []}}\n</tool_call>"}], "final_answer": "[]", "is_correct": False})
trajectories.append({"task_id": "dev_08_premature_finish", "seed": 108, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "List buckets."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"}, {"role": "tool", "content": "{\"buckets\": [\"bucket-1\"]}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"finish\", \"arguments\": {\"answer\": []}}\n</tool_call>"}], "final_answer": "[]", "is_correct": False})

# 9 & 10: Ungrounded Answer
trajectories.append({"task_id": "dev_09_ungrounded_answer", "seed": 109, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "List buckets."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"}, {"role": "tool", "content": "{\"buckets\": [\"bucket-1\"]}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"finish\", \"arguments\": {\"answer\": [\"bucket-2\"]}}\n</tool_call>"}], "final_answer": "[\"bucket-2\"]", "is_correct": False})
trajectories.append({"task_id": "dev_10_ungrounded_answer", "seed": 110, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "List IAM roles."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_iam_roles\", \"arguments\": {}}\n</tool_call>"}, {"role": "tool", "content": "{\"roles\": []}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"finish\", \"arguments\": {\"answer\": [\"phantom-role\"]}}\n</tool_call>"}], "final_answer": "[\"phantom-role\"]", "is_correct": False})

# 11 & 12: Loop Collapse
trajectories.append({"task_id": "dev_11_loop_collapse", "seed": 111, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "List buckets."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"}, {"role": "tool", "content": "{\"buckets\": [\"b1\"]}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"}, {"role": "tool", "content": "{\"buckets\": [\"b1\"]}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"}, {"role": "tool", "content": "{\"buckets\": [\"b1\"]}"}], "final_answer": "[]", "is_correct": False})
trajectories.append({"task_id": "dev_12_loop_collapse", "seed": 112, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Check policy."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_bucket_policy\", \"arguments\": {\"bucket\": \"b1\"}}\n</tool_call>"}, {"role": "tool", "content": "{}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_bucket_policy\", \"arguments\": {\"bucket\": \"b1\"}}\n</tool_call>"}, {"role": "tool", "content": "{}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_bucket_policy\", \"arguments\": {\"bucket\": \"b1\"}}\n</tool_call>"}, {"role": "tool", "content": "{}"}], "final_answer": "[]", "is_correct": False})

# 13 & 14: Redundant Call
trajectories.append({"task_id": "dev_13_redundant_call", "seed": 113, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "List buckets."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"}, {"role": "tool", "content": "{\"buckets\": [\"b1\"]}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"}, {"role": "tool", "content": "{\"buckets\": [\"b1\"]}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"finish\", \"arguments\": {\"answer\": [\"b1\"]}}\n</tool_call>"}], "final_answer": "[\"b1\"]", "is_correct": False})
trajectories.append({"task_id": "dev_14_redundant_call", "seed": 114, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Check public access."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_public_access_block\", \"arguments\": {\"bucket\": \"b1\"}}\n</tool_call>"}, {"role": "tool", "content": "{\"public\": true}"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_public_access_block\", \"arguments\": {\"bucket\": \"b1\"}}\n</tool_call>"}, {"role": "tool", "content": "{\"public\": true}"}], "final_answer": "[]", "is_correct": False})

# 15 & 16: Unrecovered Error
trajectories.append({"task_id": "dev_15_unrecovered_error", "seed": 115, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Check policy."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_bucket_policy\", \"arguments\": {\"bucket\": \"b1\"}}\n</tool_call>"}, {"role": "tool", "content": "Error: AccessDenied"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_bucket_policy\", \"arguments\": {\"bucket\": \"b1\"}}\n</tool_call>"}, {"role": "tool", "content": "Error: AccessDenied"}], "final_answer": "[]", "is_correct": False})
trajectories.append({"task_id": "dev_16_unrecovered_error", "seed": 116, "split": "dev", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Describe SG."}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"describe_security_groups\", \"arguments\": {}}\n</tool_call>"}, {"role": "tool", "content": "Error: Timeout"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"finish\", \"arguments\": {\"answer\": []}}\n</tool_call>"}], "final_answer": "[]", "is_correct": False})

# Write to file
with open("tests/fixtures/mock_trajectories_20.jsonl", "w") as f:
    for t in trajectories:
        f.write(json.dumps(t) + "\n")

print(f"Generated {len(trajectories)} trajectories at tests/fixtures/mock_trajectories_20.jsonl")