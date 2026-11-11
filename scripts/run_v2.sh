#!/usr/bin/env bash
# Build recovery data, train v2, evaluate it on the fresh block, and gate it.
#
#   ADAPTER=adapters/v0_answer2/final bash scripts/run_v2.sh
#
# The gate decides. This script does not edit a threshold, a record, or a
# decision; it only runs the pipeline that produces the candidate.
set -euo pipefail

ADAPTER="${ADAPTER:-adapters/v0_answer2/final}"
OUT="${OUT:-adapters/v2_recovery/final}"
TAG="${TAG:-v2_recovery}"
PY="${PY:-python}"

echo "== 1/4 build recovery-conditioned data =="
"$PY" -m heal.recovery --drift D1 --clean-ratio 0.5 --max-steps 24

echo "== 2/4 train v2 from ${ADAPTER} =="
"$PY" training/train_heal_recovery.py \
  --data data/heal/recovery_sft.jsonl \
  --adapter "$ADAPTER" \
  --output "$OUT" \
  --lr 5e-5 \
  --epochs 1

echo "== 3/4 evaluate on the fresh block (40000 drift / 40100 clean) =="
"$PY" -m eval.rollout \
  --adapter "$OUT" \
  --seeds 40000-40059 --bucket-label drift_targeted \
  --drift-override D1 --stale-schema --tag "$TAG"

"$PY" -m eval.rollout \
  --adapter "$OUT" \
  --seeds 40100-40159 --bucket-label gate_regression \
  --drift-override clean --tag "$TAG"

echo "== 4/4 gate against the v0 fresh baseline =="
"$PY" scripts/gate_fresh_block.py --candidate "$TAG"
