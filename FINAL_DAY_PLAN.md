# Final-day D5 experiment

**One shot. Do not rewrite the existing article around it.**
Status: candidate `v6_recovery`, D5 fail-closed drift, pre-registered thresholds
unchanged. Hard stop **13:00 IST**.

## Immutable gate blocks — never used for training

| Block | Seeds | Condition | Tag |
|---|---|---|---|
| `drift_targeted` | `70000-70199` | D5, stale schema | `v0_d5` / `v6_d5` |
| `gate_regression` | `70200-70399` | clean | `v0_d5` / `v6_d5` |

These 400 seeds are the gate only. Do not write their trajectories into
`data/drift_train`, `data/heal`, or any pair file.

## Training-only block

| Block | Seeds | Condition |
|---|---|---|
| D5 pair generation | `71000-71199` | D5, stale schema |

Only this range may feed `data/drift_train` and `data/heal/dpo_pairs.jsonl`.
Before training, assert every pair seed is in `71000-71199`:

```bash
python - <<'PY'
import json
rows = [json.loads(l) for l in open("data/heal/dpo_pairs.jsonl")]
seeds = {row["provenance"]["seed"] for row in rows}
assert seeds <= set(range(71000, 71200)), sorted(seeds - set(range(71000, 71200)))
print(f"{len(rows)} pairs, seeds {min(seeds)}..{max(seeds)}")
PY
```

## D5 contract

- `bucket` → `{}` (a valid empty payload; no error).
- `bucket_name` → real data (undocumented alias).
- Both argument names are valid to the classifier; failures are `residual`, not
  `schema_violation`.
- Pair prompt ends with the stale call and its `{}` observation, then
  chosen = `bucket_name`, rejected = continue/finish. This is the cue; without
  it the adapter learns a prior, not a conditional.

## Commands

```bash
# Gate baseline (never into data/drift_train)
python -m eval.rollout --adapter adapters/v0_answer2/final \
  --seeds 70000-70199 --bucket-label drift_targeted \
  --drift-override D5 --stale-schema --tag v0_d5
python -m eval.rollout --adapter adapters/v0_answer2/final \
  --seeds 70200-70399 --bucket-label gate_regression \
  --drift-override clean --tag v0_d5

# Training-only pairs
python -m eval.rollout --adapter adapters/v0_answer2/final \
  --seeds 71000-71199 \
  --drift-override D5 --stale-schema --tag v0_d5_train \
  --trajectories-dir data/drift_train
python -m heal.examples --drift D5 --source rollouts

# KL-DPO, beta 0.1, frozen v0 reference
python training/train_kl_dpo.py \
  --data data/heal/dpo_pairs.jsonl \
  --v0-adapter adapters/v0_answer2/final \
  --output adapters/v6_recovery --beta 0.1

# Gate candidate on the untouched blocks
python -m eval.rollout --adapter adapters/v6_recovery/final \
  --seeds 70000-70199 --bucket-label drift_targeted \
  --drift-override D5 --stale-schema --tag v6_d5
python -m eval.rollout --adapter adapters/v6_recovery/final \
  --seeds 70200-70399 --bucket-label gate_regression \
  --drift-override clean --tag v6_d5

python scripts/gate_fresh_block.py --candidate v6_d5 \
  --baseline-drift artifacts/runs/v0_d5_clean_D5_stale_drift_targeted.jsonl \
  --baseline-clean artifacts/runs/v0_d5_clean_clean_gate_regression.jsonl \
  --candidate-drift artifacts/runs/v6_d5_clean_D5_stale_drift_targeted.jsonl \
  --candidate-clean artifacts/runs/v6_d5_clean_clean_gate_regression.jsonl
```

If the gate returns `DEPLOY`, open the sealed set once:

```bash
python -m eval.rollout --adapter adapters/v6_recovery/final \
  --bucket final_sealed --sealed-ok --drift-override D5 \
  --stale-schema --tag v6_sealed
```

## Reporting

- If `DEPLOY`: add a short **Section 5b** — "final-day experiment on a
  constructed drift"; first deploy; the drift was designed after the earlier
  nulls. Do not rewrite the other sections.
- If `REJECT`: add it as one more row in the same section. Do not rewrite.
- Report task success **and** grounded success; the baseline's ~50% task success
  is the none floor.

## Hard stop

At **13:00 IST**, stop GPU work. The article, Space and repo must already be
public; D5 is additive, not a rewrite.
