# Rule Classifier Evaluation

- Trajectories evaluated: 16
- Accuracy: 1.0000
- Macro-F1 (represented classes): 1.0000
- Expected-label entries: 20
- Expected entries not present in mock file: 4

## Confusion matrix

| True \\ Predicted | env_error | schema_violation | wrong_tool | wrong_order | premature_finish | ungrounded_answer | loop_collapse | redundant_call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| env_error | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| schema_violation | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| wrong_tool | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| wrong_order | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| premature_finish | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| ungrounded_answer | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| loop_collapse | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| redundant_call | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| env_error | 1.0000 | 1.0000 | 1.0000 | 1 |
| schema_violation | 1.0000 | 1.0000 | 1.0000 | 2 |
| wrong_tool | 1.0000 | 1.0000 | 1.0000 | 2 |
| wrong_order | 1.0000 | 1.0000 | 1.0000 | 2 |
| premature_finish | 1.0000 | 1.0000 | 1.0000 | 2 |
| ungrounded_answer | 1.0000 | 1.0000 | 1.0000 | 2 |
| loop_collapse | 1.0000 | 1.0000 | 1.0000 | 2 |
| redundant_call | 1.0000 | 1.0000 | 1.0000 | 3 |

## Coverage limitations

The macro-F1 above includes only classes represented in the fixture's true or predicted labels. It is not a full 11-class validation score.

Taxonomy labels without fixture coverage: drift_mismatch, unrecovered_error, residual

Expected-label IDs absent from the mock trajectories: t1_expert, t2_random, t3_broken_schema, t4_broken_ungrounded

This is a fixture evaluation, not the final injected-drift macro-F1 measurement.
