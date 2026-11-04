# HealGuard – Evaluation Metrics Design

## 1. Overview

This document defines the five evaluation metrics used to assess agent performance, tool-use reliability, behavioral stability, and execution efficiency within the HealGuard framework.

Each metric has a predefined optimization direction to ensure consistent comparisons between baseline and healed agents.

## 2. Metric Definitions

### 2.1 Task Success (`task_success`)
- **Direction:** Maximize
- **Purpose:** Measures the proportion of trajectories in which the agent correctly completes the assigned task.
- **Formula:** Successful trajectories / Total trajectories.

### 2.2 Valid Call Rate (`valid_call_rate`)
- **Direction:** Maximize
- **Purpose:** Measures the proportion of tool calls that conform to the expected syntax and schema.
- **Formula:** Valid tool calls / Total tool calls.

### 2.3 Redundant Rate (`redundant_rate`)
- **Direction:** Minimize
- **Purpose:** Measures the proportion of unnecessary or repeated tool calls that do not contribute meaningful progress.
- **Formula:** Redundant tool calls / Total tool calls.

### 2.4 Loop Rate (`loop_rate`)
- **Direction:** Minimize
- **Purpose:** Measures the proportion of trajectories in which the agent gets stuck in repetitive execution without meaningful progress.
- **Formula:** Looping trajectories / Total trajectories.

### 2.5 Steps on Success (`steps_on_success`)
- **Direction:** Minimize
- **Purpose:** Measures the average number of execution steps required to complete successful trajectories.
- **Formula:** Total steps across successful trajectories / Number of successful trajectories.

## 3. Evaluation Summary

| Metric | Direction | Evaluation Dimension |
|---|---|---|
| `task_success` | Maximize | Task completion |
| `valid_call_rate` | Maximize | Tool-use reliability |
| `redundant_rate` | Minimize | Execution efficiency |
| `loop_rate` | Minimize | Behavioral stability |
| `steps_on_success` | Minimize | Completion efficiency |

## 4. Evaluation Considerations

- Rate-based metrics are normalized between 0 and 1.
- Empty evaluation datasets must be treated as invalid.
- Undefined denominators must be reported as not applicable.
- Redundant calls and loops must follow consistent detection criteria.
- Metric definitions and calculation rules must remain consistent across baseline and healed-agent evaluations.