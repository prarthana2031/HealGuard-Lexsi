# drift_train — failures that feed the healer

Drop v2 rollout files here. Any `*.jsonl` with `seed`, `drift` and `messages`
is picked up by `python -m heal.examples --source rollouts`.

The healer needs **failures produced by running the current agent under the
drift**, not expert traces. Each row should carry:

```json
{"task_id": "D1_6000", "seed": 6000, "drift": "D1",
 "messages": [...], "tool_calls": [...], "final_answer": []}
```

Rules that are not negotiable:

- seeds must be outside every split pool — check
  `data/splits/seed_ledger.json` before you write the file;
- a failure whose trace has **no tool call** cannot produce a minimal-edit DPO
  pair. That is expected before v0 can call tools, and `heal.examples` says so
  instead of inventing a pair.

The world v1 rollouts that used to live here are in
[`data/archive/world_v1/`](../archive/world_v1/README.md) — same seed numbers,
different world, do not mix them in.
