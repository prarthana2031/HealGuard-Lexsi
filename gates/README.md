# HealGuard Deployment Gate

## 1. Purpose

Evaluates whether a candidate agent is eligible for deployment compared with
its baseline using the frozen thresholds in `configs/thresholds.yaml`.

The gate follows a fail-closed approach: insufficient evidence results in
BLOCKED, never an assumed PASS.

## 2. Ownership

- Gate implementation and documentation: Person C
- Model training and pipeline integration: Person A
- Expert trajectories and oracle data: Person B

## 3. Files

- `deployment_gate.py`: Deployment checks and comparison logic.
- `thresholds.py`: Loads and validates frozen thresholds.
- `test_deployment_gate.py`: Synthetic unit tests.
- `README.md`: Documentation and integration instructions.

## 4. Frozen Deployment Criteria

| Criterion | Requirement |
|---|---:|
| Minimum observations per evaluation bucket | 40 |
| Task-success gain | At least +0.10 |
| Statistical significance | One-sided paired bootstrap, alpha = 0.05 |
| Worst-family success drop | At most 0.05 |
| Median steps on success | At most 1.05 times baseline |
| Redundant-call rate | Must not increase |
| New failure-mode share | No label newly above 0.02 |

Thresholds are loaded from `configs/thresholds.yaml`.
The shared configuration must remain unchanged.

## 5. Evaluation Buckets

The gate evaluates two buckets separately:

- `drift_targeted`
- `gate_regression`

Each bucket requires at least 40 paired task observations.

The minimum sample requirement applies to the total bucket size,
not separately to F1, F2 and F4.

## 6. Resource Families

- F1: S3 buckets.
- F2: IAM roles.
- F4: Security groups.

Worst-family regression is evaluated separately when a family has at least
10 observations. Families below this count are skipped for the individual
family check. Overall bucket success remains the primary evaluation.

## 7. Input Contract

Each evaluation record must provide:

- `id` or `task_id`: Unique task identifier.
- `split` or `bucket`: Evaluation bucket.
- `success`: Boolean task outcome.
- `num_steps`: Non-negative integer.
- `total_tool_calls`: Non-negative integer.
- `redundant_calls`: Non-negative integer.
- `family` or `task_family`: F1, F2 or F4.
- `failure_mode`: Primary classifier label for unsuccessful trajectories.

Baseline and candidate records must have identical task IDs within each bucket.

These are evaluation-summary fields. They do not modify
`agent/trajectory_schema.md`.

## 8. New Failure-Mode Rule

For each failure label, calculate:

Failure-label share = Number of trajectories with that label / Total
candidate trajectories in the evaluation bucket.

A label is flagged when its share crosses from at most 2% in the baseline
to above 2% in the candidate.

The denominator is all trajectories in the bucket, not only failed
trajectories.

Missing failure labels block the check.

## 9. Deployment Checks

The gain is gated **only on the targeted bucket**. The regression bucket has to
*not get worse*; requiring a gain there would reject a candidate that fixed the
drift and left clean behaviour unchanged, which is the normal outcome.

| Check | `drift_targeted` | `gate_regression` |
|---|---|---|
| Paired task coverage | gate | gate |
| Minimum 40 observations | gate | gate |
| Task-success gain ≥ 0.10 | **gate** | reported, not gated |
| Paired-bootstrap lower bound > 0 | **gate** | reported, not gated |
| Worst-family drop ≤ 0.05 | gate | **gate** |
| Median successful steps ≤ 1.05× | gate | gate |
| Redundant-call rate not higher | gate | gate |
| No label newly above 2% | gate | gate |

### Efficiency is compared on grounded successes

An answer can be right by luck: on a `none` task, `[]` is correct even if the
agent never read anything. Measured on the v0 rollouts under D1, **32 of 37
successes (86%) were ungrounded**, and they took a median of **12 steps against
14** for the grounded ones. A step comparison over all successes therefore
rewards luck and punishes work.

The population is now tasks where **both** runs were correct *and* made every
required evidence call (`world/evidence.py`). Thresholds are unchanged, including
the 1.05x limit — only who is in the sample changed. This rule was revised after
seeing the first result; it is disclosed as post-hoc, and the legacy number is
reported in the same check value under `legacy_*` so both are visible.

Records from before the change carry no `grounded_success` field; for those the
gate falls back to the old population and labels it "grounding not recorded"
rather than assuming they are ungrounded.

`steps_on_success` and `redundant_call_rate` are computed on the tasks **both
runs solved**, and on the **clean-API subset** when the bucket has one.

Why: on a drifted task a broken baseline often succeeds *because* it did less
work. It could not read the listing, so it skipped every resource and happened
to answer `[]`, which is correct for a `none` task. Counting that as "more
efficient" rejects a healer for doing the reads it now can do. The unpaired
medians are still reported in the check value, alongside `compared_on`, so
nothing is hidden.

Bootstrap configuration:
- Iterations: 10,000.
- Random seed: 42.
- Alpha: 0.05.

## 10. Result Status

Internal status, and the published outcome it maps to:

| Status | Published | Meaning |
|---|---|---|
| `PASS` | `DEPLOY` | All checks meet their requirements. |
| `FAIL` | `REJECT` | At least one measured criterion violates its threshold. `failed_checks` names them. |
| `BLOCKED` | `NEED_MORE_DATA` | Evidence missing, sample counts insufficient, or task coverage differs. |

Every outcome is appended to `artifacts/decision_log.jsonl` with every number,
the frozen thresholds, and a human-readable `reason`.

## 11. Usage

```python
from gates.deployment_gate import evaluate_deployment_gate

result = evaluate_deployment_gate(
    baseline_records,
    candidate_records
)

print(result["status"])
print(result["buckets"])