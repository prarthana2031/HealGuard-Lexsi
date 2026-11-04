
# Trajectory Format Contract (Person A → B & C)

## Required top-level fields

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Unique trajectory id |
| `seed` | int | Random seed used |
| `split` | string | one of: train_clean, dev, drift_train, drift_targeted, gate_regression, final_sealed |
| `task` | string | The original user question / task |
| `messages` | list[dict] | Full chat history (system / user / assistant / tool) |
| `tool_calls` | list[dict] | Parsed tool calls in order |
| `reward` | float | 1.0 = success, 0.0 = failure |
| `success` | bool | Same as reward == 1.0 |
| `num_steps` | int | Number of assistant turns / tool calls |
| `template_id` | string | Which chat template was used |
| `error` | string or null | Last error message if any, else null |
| `valid_tool_calls` | int | How many tool calls were syntactically + schema-valid |
| `total_tool_calls` | int | Total tool calls attempted |
| `redundant_calls` | int | Repeated / unnecessary tool calls |
| `finished` | bool | Whether the agent produced a final answer |

## Example trajectory (minimal)

```json
{
  "id": "traj-00042",
  "seed": 42,
  "split": "dev",
  "task": "Is bucket public-read allowed?",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Is bucket public-read allowed?"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"check_bucket_acl\", \"arguments\": {\"bucket\": \"my-bucket\"}}\n</tool_call>"},
    {"role": "tool", "content": "{\"public\": false}"},
    {"role": "assistant", "content": "No, the bucket is not public-read."}
  ],
  "tool_calls": [
    {
      "name": "check_bucket_acl",
      "arguments": {"bucket": "my-bucket"},
      "valid": true
    }
  ],
  "reward": 1.0,
  "success": true,
  "num_steps": 2,
  "template_id": "custom_tool_call_v1",
  "error": null,
  "valid_tool_calls": 1,
  "total_tool_calls": 1,
  "redundant_calls": 0,
  "finished": true
}