# HealGuard – Failure Classifier Design

## 1. Objective

The failure classifier identifies and categorizes agent execution failures using deterministic, rule-based logic.

It uses the predefined eleven-label failure taxonomy to assign one primary failure category to each evaluated trajectory.

## 2. Classification Architecture

The classifier follows three stages:

1. **Evidence Extraction:** Collect relevant execution information, including environment errors, tool calls, execution order, observations, and termination status.
2. **Rule Evaluation:** Evaluate failure conditions according to the predefined taxonomy priority.
3. **Label Assignment:** Assign the first matching failure category to the trajectory.

## 3. Classification Priority

The classifier must follow the priority order defined in `taxonomy.md`.

When multiple failure conditions are detected, the highest-priority matching category takes precedence.

## 4. Classification Constraints

- Classification must be deterministic and rule-based.
- Each trajectory receives exactly one primary failure label.
- Classification must not depend on LLM-generated judgments.
- Residual failures must be counted without being sent to an LLM.
- Missing or insufficient evidence must not be treated as proof of a failure category.

## 5. Implementation Requirements

- Implement classification rules as independent, testable conditions.
- Separate evidence extraction from label assignment.
- Maintain strict priority ordering.
- Support reproducible classification results.
- Validate classification behavior using representative test trajectories.

## 6. Integration

The classifier will consume the finalized trajectory representation and produce a primary failure label for downstream failure analysis and evaluation.