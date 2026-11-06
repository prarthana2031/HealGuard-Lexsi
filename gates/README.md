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

1. Paired task coverage.
2. Minimum 40 observations in each evaluation bucket.
3. Task-success improvement of at least 0.10.
4. One-sided paired-bootstrap lower confidence bound greater than zero.
5. Worst-family regression no greater than 0.05.
6. Median successful steps no greater than 1.05 times baseline.
7. Redundant-call rate must not increase.
8. No failure label may newly exceed 2% share.

Bootstrap configuration:
- Iterations: 10,000.
- Random seed: 42.
- Alpha: 0.05.

## 10. Result Status

- `PASS`: All checks meet their requirements.
- `FAIL`: At least one measured criterion violates its threshold.
- `BLOCKED`: Required evidence is missing, sample counts are insufficient,
  or task coverage differs.

## 11. Usage

```python
from gates.deployment_gate import evaluate_deployment_gate

result = evaluate_deployment_gate(
    baseline_records,
    candidate_records
)

print(result["status"])
print(result["buckets"])