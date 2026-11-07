# drift_train — failures that feed the healer

Drop v2 rollout files here. Any `*.jsonl` carrying `seed`, `drift` and
`messages` is picked up by `python -m heal.examples --source rollouts`.

## Generate them directly here

```bash
python -m eval.rollout \
    --adapter adapters/v0/final \
    --bucket drift_targeted \
    --tag v0_d1 \
    --prompt-variant clean \
    --trajectories-dir data/drift_train
```

That writes two files:

- `artifacts/runs/v0_d1_drift_targeted.jsonl` — gate-ready records
- `data/drift_train/v0_d1_drift_targeted.jsonl` — full trajectories for the healer

## Then build the examples

```bash
python -m heal.examples --drift D1 --source rollouts
```

## Rules that are not negotiable

- **Use `--prompt-variant clean`.** A prompt-hinted failure starts from a
  different system prompt than the expert's, so no minimal-edit pair can be
  built from it. `heal.examples` refuses rather than forcing one. The
  prompt-only arm exists as a *control*, not as training data.
- **Seeds must be outside every split pool.** `drift_targeted` is 6000–6075 and
  `gate_regression` is 10000–10065, both frozen. Generate rollouts on a fresh
  block — 30000–30059 is reserved for this — and keep
  `data/splits/seed_ledger.json` in step with it.
- **A trace with no tool call cannot produce a pair.** That is expected until v0
  can call tools; `heal.examples` reports how many it skipped instead of
  inventing one.

The world v1 rollouts that used to live here are in
[`data/archive/world_v1/`](../archive/world_v1/README.md) — same seed numbers,
different world, do not mix them in.
