# expert — the scripted teacher

`run_expert(snapshot, inject_error=False, drift=None)` returns one gold
trajectory in the exact format the trainer and the scorers consume.

It is the teacher for v0 SFT. The plan forbids distilling from a larger LLM, so
this file is the whole "teacher" story.

## What it guarantees

- **100% oracle accuracy**, on clean and under every drift (asserted in
  `tests/test_world.py`).
- **Recovery traces that actually recover.** The 503 branch retries with a
  *changed* call (`limit=5`). The previous version repeated the identical call,
  which taught the student that repeating a failing call is the correct
  behaviour — the exact failure the project exists to fix.
- **Per-family success**, so the gate can build a per-family regression table
  instead of a blended mean that hides a collapse.

## Argument names follow the API version

| Drift | What the expert has to do |
|---|---|
| D1 | pass `bucket_name` instead of `bucket` |
| D2 | read `bucket_list` / `continuation` |
| D3 | pass `region` on `list_buckets` and `describe_security_groups` |
| D4 | call `get_acl` instead of `get_bucket_acl` |

This is allowed: the expert is the teacher under the *new* schema. The student is
not.
