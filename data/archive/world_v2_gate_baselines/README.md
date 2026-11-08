# Archived: gate baseline rollouts

`v0_drift_targeted.jsonl` - the v0_answer2 run on the `drift_targeted` bucket
(seeds 6000-6075), used to produce `artifacts/runs/v0_drift_targeted.jsonl`, the
gate's baseline records.

## Why it is here and not in `data/drift_train/`

54 of its 60 seeds are `drift_targeted` tasks and **all 11 of its failures** are
in seeds 6001-6050 - the exact tasks the gate grades. It sat in `drift_train/`,
where the healer looks for training failures, so the first attempt to build DPO
pairs from it would have trained v1_good on the gate's own tasks and manufactured
the gate's gain.

`heal.examples` now refuses any rollout whose seed belongs to an evaluation
split, and it would have caught this. Moving the file here means the refusal is
not the only thing standing between us and that mistake.

Keep it: it is the baseline half of the gate comparison. Do not train on it.
