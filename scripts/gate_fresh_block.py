"""Gate a candidate against the v0 fresh block, with explicit file paths.

`eval.rollout --compare-to` assumes one prefix for both buckets.  The fresh
block was written under two labels (`v0_fresh_clean_D1_stale` for the drift
bucket and `v0_fresh_clean` for the clean bucket), so that convenience path
cannot find the baseline.  This runner loads the real files, calls the same
frozen gate, prints every check, and appends the decision to the log.

The candidate must be evaluated on the same fresh block:

    python -m eval.rollout --adapter adapters/v2_recovery/final \
        --seeds 40000-40059 --bucket-label drift_targeted \
        --drift-override D1 --stale-schema --tag v2_recovery

    python -m eval.rollout --adapter adapters/v2_recovery/final \
        --seeds 40100-40159 --bucket-label gate_regression \
        --drift-override clean --tag v2_recovery

Then:

    python scripts/gate_fresh_block.py --candidate v2_recovery

This script does not change a threshold, a record, or a decision.  If the gate
says REJECT, the answer is no.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gates.deployment_gate import evaluate_deployment_gate  # noqa: E402
from gates.registry import log_decision  # noqa: E402

RUNS = ROOT / "artifacts" / "runs"
DEFAULT_BASELINE_DRIFT = RUNS / "v0_fresh_clean_D1_stale_drift_targeted.jsonl"
DEFAULT_BASELINE_CLEAN = RUNS / "v0_fresh_clean_clean_gate_regression.jsonl"


def load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing records: {path}")
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows:
        raise SystemExit(f"empty records: {path}")
    return rows


def print_bucket(name: str, bucket: dict[str, Any]) -> None:
    print(f"\n  {name}: {bucket['status']}")
    for check, payload in bucket["checks"].items():
        value = payload.get("value")
        if isinstance(value, dict):
            compact = ", ".join(f"{k}={v}" for k, v in list(value.items())[:4])
            value = "{" + compact + "}"
        print(f"    {check:32s} {payload['status']:8s} {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="tag used by eval.rollout")
    parser.add_argument("--candidate-drift", type=Path)
    parser.add_argument("--candidate-clean", type=Path)
    parser.add_argument("--baseline-drift", type=Path, default=DEFAULT_BASELINE_DRIFT)
    parser.add_argument("--baseline-clean", type=Path, default=DEFAULT_BASELINE_CLEAN)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    candidate_drift = args.candidate_drift or RUNS / f"{args.candidate}_drift_targeted.jsonl"
    candidate_clean = args.candidate_clean or RUNS / f"{args.candidate}_gate_regression.jsonl"

    baseline = load(args.baseline_drift) + load(args.baseline_clean)
    candidate = load(candidate_drift) + load(candidate_clean)

    print(f"baseline:  {args.baseline_drift.name} + {args.baseline_clean.name}")
    print(f"candidate: {candidate_drift.name} + {candidate_clean.name}")

    result = evaluate_deployment_gate(baseline, candidate)

    for name, bucket in result["buckets"].items():
        print_bucket(name, bucket)

    print(f"\n  decision: {result['decision']}")
    print(f"  reason:   {result['reason']}")

    if not args.no_log:
        record = log_decision(
            result,
            baseline=f"{args.baseline_drift.name} + {args.baseline_clean.name}",
            candidate=args.candidate,
            meta={
                "candidate_drift": str(candidate_drift),
                "candidate_clean": str(candidate_clean),
                "baseline_drift": str(args.baseline_drift),
                "baseline_clean": str(args.baseline_clean),
            },
        )
        print(f"  logged:   {record['logged_at']} -> artifacts/decision_log.jsonl")

    # A non-zero exit on REJECT makes a CI step honest; scripts that just want
    # the printout can pass --no-log or ignore the code.
    raise SystemExit(0 if result.get("decision") == "DEPLOY" else 2)


if __name__ == "__main__":
    main()
