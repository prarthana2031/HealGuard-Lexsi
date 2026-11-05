# HealGuard - Evaluation Metrics Design

## 1. Overview

This document defines the five evaluation metrics used to assess agent performance, tool-use reliability, behavioral stability, and execution efficiency.

All metrics follow the trajectory contract in `agent/trajectory_schema.md`.

## 2. Metric Definitions

### 2.1 Task Success (`task_success`)

- Direction: Maximize
- Purpose: Measures the proportion of trajectories in which the agent successfully completes the assigned task.
- Input: `success` (boolean)
- Formula: Successful trajectories / Total trajectories.
- Aggregation: Mean success across all trajectories.

### 2.2 Valid Call Rate (`valid_call_rate`)

- Direction: Maximize
- Purpose: Measures the proportion of attempted tool calls that are syntactically and schema-valid.
- Inputs: `valid_tool_calls`, `total_tool_calls`
- Formula: Valid tool calls / Total tool calls.
- Edge case: N/A when no tool calls were attempted.

### 2.3 Redundant Rate (`redundant_rate`)

- Direction: Minimize
- Purpose: Measures the proportion of unnecessary or repeated tool calls.
- Inputs: `redundant_calls`, `total_tool_calls`
- Formula: Redundant tool calls / Total tool calls.
- Edge case: N/A when no tool calls were attempted.

### 2.4 Loop Rate (`loop_rate`)

- Direction: Minimize
- Purpose: Measures the proportion of trajectories in which the agent gets stuck in repetitive execution without meaningful progress.
- Inputs: `messages`, `tool_calls`
- Detection: AuditKIT `AgentLoopDetection`.
- Configuration:
  - repetition_threshold: 3
  - similarity_threshold: 0.85
- Formula: Looping trajectories / Total trajectories.
- Loop detection is performed at evaluation time. No additional loop field is required in the trajectory contract.

### 2.5 Steps on Success (`steps_on_success`)

- Direction: Minimize
- Purpose: Measures execution steps required to complete successful trajectories.
- Inputs: `success`, `num_steps`
- Per-trajectory value: `num_steps` for successful trajectories only.
- Dataset aggregation: Median number of steps across successful trajectories.
- Unsuccessful trajectories are excluded.
- Edge case: N/A when there are no successful trajectories.

## 3. Evaluation Summary

| Metric | Direction | Evaluation Dimension |
|---|---|---|
| task_success | Maximize | Task completion |
| valid_call_rate | Maximize | Tool-use reliability |
| redundant_rate | Minimize | Execution efficiency |
| loop_rate | Minimize | Behavioral stability |
| steps_on_success | Minimize | Completion efficiency |

## 4. Evaluation Considerations

- Rate-based metrics are normalized between 0 and 1.
- Empty evaluation datasets are invalid.
- Undefined denominators must be reported as N/A, not zero.
- Valid and redundant call rates are N/A when total tool calls are zero.
- Steps on success is N/A when no trajectories succeed.
- Loop detection uses AuditKIT at evaluation time.
- Redundant calls and loops use consistent detection criteria.
- The same metric definitions must be used for baseline and healed-agent evaluations.
- The original trajectory data must not be modified to manufacture metric values.

## 5. Trajectory Contract Mapping

| Metric | Required trajectory fields |
|---|---|
| task_success | success |
| valid_call_rate | valid_tool_calls, total_tool_calls |
| redundant_rate | redundant_calls, total_tool_calls |
| loop_rate | messages, tool_calls |
| steps_on_success | success, num_steps |

## 6. Implementation

The five scorers are implemented in `eval/metrics.py`.

Dataset-level aggregation is implemented in `eval/auditkit_runner.py`.

Unit tests are maintained in:
- `eval/test_metrics.py`
- `eval/test_auditkit_runner.py`
