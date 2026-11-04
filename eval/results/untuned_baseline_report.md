# Untuned Tiny Aya Baseline — 5 November 2026

## Evaluation setup

- Model: CohereLabs/tiny-aya-global
- Model state: Untuned base model (no LoRA adapter)
- Dataset: 20 development trajectories
- Evaluation framework: AuditKIT
- Evaluation runner: `eval/auditkit_runner.py`

## Dataset verification

- Total trajectories: 20
- Unique task IDs: 20
- Successful trajectories: 0
- Failed trajectories: 20
- Assistant tool-call tags: 0
- Tool execution messages: 0

## Official baseline metrics

| Metric | Result | Numerator | Denominator |
|---|---:|---:|---:|
| Task success | 0.0 | 0 | 20 |
| Valid call rate | N/A | 0 | 0 |
| Redundant call rate | N/A | 0 | 0 |
| Loop rate | 0.0 | 0 | 20 |
| Steps on success | N/A | 0 | 0 |

## Interpretation

The untuned model achieved zero successful trajectories across the 20 development tasks.

No structured tool calls were generated. Therefore, valid call rate and redundant call rate are not applicable because no tool calls were made.

No loops were detected across the 20 trajectories. Steps on success is not applicable because there were no successful trajectories.

## Output files

- `eval/results/untuned_baseline.jsonl` — Original baseline trajectories.
- `eval/results/untuned_baseline_adapted.jsonl` — Adapted trajectories used by the AuditKIT runner.

## Verification

- All 20 records were successfully adapted.
- Dataset-level AuditKIT evaluation completed.
- Full test suite: 18 tests passed.
