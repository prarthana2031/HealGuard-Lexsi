# Trajectory Format Contract (Person A → B & C)

Every trajectory must contain:

- `id`: unique string
- `seed`: int
- `split`: one of [train_clean, dev, drift_train, drift_targeted, gate_regression, final_sealed]
- `messages`: list of dicts with roles: system / user / assistant / tool
- `tool_calls`: list of parsed tool calls (name + arguments)
- `reward`: float (1.0 = success)
- `success`: bool
- `num_steps`: int
- `template_id`: string (which chat template was used)

Tool call format we train & evaluate on:
<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>