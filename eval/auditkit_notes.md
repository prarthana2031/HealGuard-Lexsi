# AuditKIT Source Review Notes

## 1. Package Information

- Package: AuditKIT
- Installed inside the HealGuard project virtual environment.
- Source location: `.venv/Lib/site-packages/auditkit/`
- Reviewed source files:
  - `api.py`
  - `comparison.py`
  - `metrics/agent.py`

## 2. Evaluation API - `api.py`

### `ak.evaluate()`

- Main entry point for running an evaluation.
- Accepts a dataset, model, scorers, and optional configuration.
- Supports two engines:
  - `native` (default): runs the Adapter -> Model -> Metric pipeline.
  - `lmeval`: uses lm-evaluation-harness.
- Converts the dataset into a scenario.
- Resolves the model, adapter, and metrics.
- Executes evaluation through `Runner().run()`.
- Returns a `RunResult`.

HealGuard relevance:
- The native engine supports custom scorers.
- HealGuard can use its own trajectory adapter and evaluation metrics.

### `ak.compare()`

Supports two comparison modes:

1. Two-run comparison:
   `ak.compare(baseline_run, candidate_run)`

   Returns a `RunComparison` containing per-task deltas and comparison diagnostics.

2. Multiple-run leaderboard:
   `ak.compare([run1, run2], metric="metric_name")`

   Returns run summaries sorted by the selected metric.

HealGuard relevance:
- Can compare baseline and healed-agent evaluation runs.
- Can support leaderboard comparisons across multiple runs.

## 3. Failure Diagnostics - `comparison.py`

### `failure_summary()`

Returns:
- `baseline_failed`
- `candidate_failed`
- `baseline_errors`
- `candidate_errors`

Important behavior:
- Samples that fail during annotation or scoring can have empty score metadata.
- Such samples may be excluded from metric means rather than counted as incorrect.
- This can make a run appear better because its effective denominator has decreased.

### `has_failures`

Returns whether either the baseline or candidate run contains failed samples.

### `coverage_warnings()`

Checks sample-count differences:
- Across each metric overall.
- Across each task and metric separately.

A mismatch means baseline and candidate metrics may have been calculated over different sample sets.

Important:
- Overall counts can match while individual task counts differ.
- A comparison with coverage mismatches should not be treated as a fully like-for-like comparison.

HealGuard evaluation practice:
- Inspect `has_failures`.
- Inspect `failure_summary()`.
- Inspect `coverage_warnings()`.
- Review these diagnostics before interpreting metric improvements.

## 4. Loop Detection - `metrics/agent.py`

### `AgentLoopDetection`

- Metric name: `agent_loop_detection`
- Kind: `AGENT`
- Direction: `MAXIMIZE`
- Reference-free and deterministic.
- Does not require an LLM judge or reference trajectory.

Constructor defaults:
- `repetition_threshold = 3`
- `similarity_threshold = 0.85`

The constructor rejects repetition thresholds below 2.

### Three loop signals

1. Repeated calls:
   - Detects identical tool name and arguments.
   - Flags when the repetition count reaches the configured threshold.

2. Reasoning stagnation:
   - Compares consecutive assistant/reasoning messages.
   - Uses `difflib.SequenceMatcher`.
   - Flags similarity at or above the configured threshold.
   - Differences explained only by tool-call arguments are excluded from this stagnation signal.

3. Call-graph cycle:
   - Detects a cycle in the tool-call sequence.
   - Uses DFS to identify a back edge.

### Output behavior

- Score `1.0`: no loop detected.
- Score `0.0`: at least one loop signal detected.
- Returns no score when there are no calls and no assistant messages.

The score includes:
- `looped`
- `signals`
- `max_repeat`
- `n_calls`

### HealGuard integration

HealGuard uses:
- Repetition threshold: 3
- Similarity threshold: 0.85

The AuditKIT detector is used at evaluation time. The trajectory contract does not need an additional loop field.

HealGuard's `loop_rate` is the proportion of trajectories identified as looping. This is the inverse interpretation of AuditKIT's loop-free score.

## 5. Important Evaluation Safeguards

1. Failed samples must not be silently ignored when interpreting evaluation results.
2. Baseline and candidate sample coverage must be checked.
3. Per-task coverage must be checked, not only overall coverage.
4. Loop detection must use the same thresholds across comparable runs.
5. Baseline and healed-agent evaluations must use consistent metric definitions.
6. Raw trajectory evidence must remain unchanged.

## 6. Source Review Summary

The reviewed AuditKIT functionality provides:

- `evaluate()` for running evaluations.
- `compare()` for comparing runs or building leaderboards.
- `failure_summary()` for failed-sample diagnostics.
- `coverage_warnings()` for sample coverage checks.
- `AgentLoopDetection` for deterministic loop detection.

These capabilities support HealGuard's evaluation and comparison workflow.
