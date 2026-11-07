# Archived: world v1 observations

`v0_failed_rollouts.jsonl` — 80 real rollouts from the **untuned**
`CohereLabs/tiny-aya-global`, run under D1–D4 against **world v1** snapshots.

## Why it is here and not in `data/drift_train/`

World v2 changed the snapshot generator, so seed 2000 is now a different cloud
account. These traces no longer correspond to any task the pipeline can score.
Keeping them in a live split would be worse than deleting them: a healer built on
them would learn to fix an account that does not exist.

## What they are still good for

The **classifier** reads traces, not worlds. These 80 failures are the only real
model evidence in the repo, and they demonstrate the thing that mattered most:
the previous classifier returned `residual` for all 80, while the current one
returns `schema_violation` (correctly — the untuned model emits prose, so there
is no tool call whose name or argument could be in a schema diff).

`classifier/evaluate_classifier.py` scores them as the "field" number, labelled
as world v1. Do not quote them as v2 results, and do not put them in a v2 gate
run.

## Not deleted, on purpose

Deleting them would erase the evidence that the probe was done. They are small,
they are labelled, and they are outside every split and every seed pool
(seeds 2000–2019 are reserved in `data/splits/seed_ledger.json`).
