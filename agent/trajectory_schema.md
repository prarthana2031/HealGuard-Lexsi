# Trajectory Format Contract

One shape, used by the expert, the rollout harness and the healer. `agent/schema.py`
mirrors this file; `tests/test_no_leakage.py` asserts that every split on disk
carries the fields the scorers read.

## Required top-level fields

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Unique trajectory id, `"{drift or clean}_{seed}"` |
| `seed` | int | Snapshot seed |
| `split` | string | `train_clean`, `dev`, `drift_train`, `drift_targeted`, `gate_regression`, `final_sealed`, or `rollout` |
| `task` | string | The user question |
| `messages` | list[dict] | Full chat history (`system` / `user` / `assistant` / `tool`) |
| `tool_calls` | list[dict] | Parsed calls, in order, `{"name": ..., "arguments": {...}}` |
| `final_answer` | list[str] | The set submitted at `finish` |
| `reward` | float | `heal.replay` reward: correctness × step penalty + capped shaping |
| `success` | bool | Exact set match against the oracle |
| `num_steps` | int | Tool calls before `finish` |
| `template_id` | string | Chat template used, e.g. `custom_tool_call_v1` |
| `valid_tool_calls` | int | Calls that did not raise or return an error payload |
| `total_tool_calls` | int | Calls attempted, including `finish` |
| `redundant_calls` | int | Identical `(tool, args)` repeats |
| `finished` | bool | Whether the episode ended with `finish` |

## Optional fields

| Field | Type | Meaning |
|-------|------|---------|
| `drift` | string or null | API version the run faced: `D1`–`D4`, or `null` for clean |
| `drift_override` | string or null | Set by `eval.rollout --drift-override` |
| `prompt_variant` | string | `clean` or a `prompt_only_*` control arm |
| `required_tools` | list[str] | Expert traces only; what a correct run must call |
| `family_success` | dict[str, bool] | `{"F1": ..., "F2": ..., "F4": ...}` exact-match per family |
| `answer_kind` | string | `none` or `vulnerable`, set by the split builder |
| `is_none_task` | bool | True when the oracle's answer is the empty set |
| `invalid_calls` | int | Rollouts only: responses that contained no parseable call |
| `error` | string or null | Last error, else null |

## Example trajectory (minimal, truncated)

```json
{
  "id": "D1_6000",
  "seed": 6000,
  "split": "drift_targeted",
  "drift": "D1",
  "task": "Find all vulnerable cloud resources.",
  "template_id": "custom_tool_call_v1",
  "messages": [
    {"role": "system", "content": "...tool list..."},
    {"role": "user", "content": "Find all vulnerable cloud resources."},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"list_buckets\", \"arguments\": {}}\n</tool_call>"},
    {"role": "tool", "content": "{\"items\": [\"bucket-6000-0\"], \"next_token\": null}"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_bucket_acl\", \"arguments\": {\"bucket_name\": \"bucket-6000-0\"}}\n</tool_call>"},
    {"role": "tool", "content": "{\"acl_public\": true}"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"finish\", \"arguments\": {\"answer\": [\"bucket-6000-0\"]}}\n</tool_call>"}
  ],
  "tool_calls": [
    {"name": "list_buckets", "arguments": {}},
    {"name": "get_bucket_acl", "arguments": {"bucket_name": "bucket-6000-0"}},
    {"name": "finish", "arguments": {"answer": ["bucket-6000-0"]}}
  ],
  "final_answer": ["bucket-6000-0"],
  "family_success": {"F1": true, "F2": true, "F4": true},
  "reward": 0.97,
  "success": true,
  "num_steps": 2,
  "valid_tool_calls": 3,
  "total_tool_calls": 3,
  "redundant_calls": 0,
  "finished": true,
  "error": null
}
```

The example above is a *drifted* run: the system prompt advertises `bucket_name`,
which is what the agent is shown. The student that fails this task is the one
that still emits `bucket`.
