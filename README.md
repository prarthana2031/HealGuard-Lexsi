## Pinned versions (frozen 3 Nov)

- Python: 3.12.x
- AgentTune: `d4db4c12d52342227c4113a70183d0f64aa2f345`
- AuditKIT: `71316da345784ff220d5ed59337441a59ab31526`

## Frozen Gate Thresholds (written 3 Nov, frozen 11 Nov)

- min_n per bucket: 40
- min gain on task_success: +0.10 absolute
- statistical test: paired bootstrap, α = 0.05 (one-sided)
- worst-family drop: ≤ 0.05
- steps on success: median ≤ 1.05 × previous model
- redundant-call rate: not higher than previous model

## S2 Results (3 Nov – Colab T4)

- Template used: Custom TRL-style `<tool_call>`
- Stock template tool call: NO
- Custom template tool call: YES
- Generation time: 6.1 s
- Peak GPU memory: 5.13 GB
- Trainable params: 30.2M (0.89%)