# HealGuard Failure Taxonomy

## 1. Overview

This taxonomy defines the failure categories used to identify and classify agent failures during execution and evaluation.

Each trajectory is assigned a single primary failure label based on the priority order defined below.

## 2. Failure Categories

| Priority | Failure Label | Description |
|---:|---|---|
| 1 | `env_error` | Failure caused by an environment or execution error. |
| 2 | `drift_mismatch` | Agent behavior is incompatible with the current environment or tool configuration. |
| 3 | `schema_violation` | Tool call violates the expected input or output schema. |
| 4 | `wrong_tool` | Agent selects an incorrect tool for the required operation. |
| 5 | `wrong_order` | Agent executes operations in an incorrect sequence. |
| 6 | `premature_finish` | Agent terminates execution before completing the task. |
| 7 | `ungrounded_answer` | Agent produces an answer unsupported by available observations or tool results. |
| 8 | `loop_collapse` | Agent becomes trapped in repetitive execution without meaningful progress. |
| 9 | `redundant_call` | Agent makes unnecessary or repeated tool calls. |
| 10 | `unrecovered_error` | Agent fails to recover after encountering an error. |
| 11 | `residual` | Failure does not match any of the preceding categories. |

## 3. Classification Priority

The following priority order is mandatory:

1. env_error
2. drift_mismatch
3. schema_violation
4. wrong_tool
5. wrong_order
6. premature_finish
7. ungrounded_answer
8. loop_collapse
9. redundant_call
10. unrecovered_error
11. residual

## 4. Classification Rules

- Classification must be rule-based.
- The first matching category in the priority order takes precedence.
- Each trajectory receives exactly one primary failure label.
- The priority order must remain consistent across evaluation runs.
- `residual` serves as the fallback category for failures not covered by the first ten labels.
- Residual failures must only be counted and must never be sent to an LLM.