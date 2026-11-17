# HealGuard: A Verifiable Deployment Gate for Self-Healing Tool-Calling Agents

**One-line claim:** We built a pre-registered gate that rejected every heal we
tried — including a +10-point fix — because each one traded a visible failure
for a hidden one.

**Track:** Track 3 — Agentic Fine-Tuning (single Track 3 problem statement; no
sub-number is assigned in the repo). **Content warning:** none.

[Article](https://huggingface.co/bononi/healguard-article) ·
[Models](https://huggingface.co/collections/bononi/healguard-models) ·
[Dataset](https://huggingface.co/datasets/bononi/healguard-data) ·
[Space](https://huggingface.co/spaces/bononi/healguard-timeline) ·
[Code](https://github.com/prarthana2031/HealGuard-Lexsi)

## Abstract

No model was deployed. We built a deterministic cloud-audit world, five API
drifts, a replay-verified healing pipeline, and a pre-registered deployment
gate, then tested ten candidate entries: the baseline, eight candidate models,
and a runtime guard. Every one was rejected.

**Finding 1 — the none-floor illusion.** The baseline looked like a 75% agent
but was a **35% grounded** agent; half the benchmark is `none`, so answering
`[]` is free. On the final 200-task D5 block, v0 scored **56.5% raw / 36.5%
grounded** on drift and **80.0% raw / 45.0% grounded** on clean.

**Finding 2 — the guard side-effect.** A runtime repair raised D1 raw success
from **75% to 85%** and grounded from **35% to 58.3%**, leaving clean at
**73.3%**, but the gate rejected it for **11.7% `residual`** plus **3.3%
`premature_finish`** new failure modes. A success-only dashboard would have
shipped it.

**Headline before → after:** v0 **75.0% raw / 35.0% grounded** → Runtime Guard
**85.0% raw / 58.3% grounded**; gate **REJECT** (11.7% residual, 3.3% premature).

**Base model + method:** `CohereLabs/tiny-aya-global` (3.35B, 4-bit nf4, LoRA
r=16/α=32), SFT/DPO/KL-DPO on replay-verified traces, evaluated by HealGate.

**For judges:** read the two tables under *Results*; the first is the
raw/grounded gap, the second is the guard passing every aggregate check and
failing the new-failure-mode check. Full write-up: [`ARTICLE.md`](ARTICLE.md);
standalone abstract: [`ABSTRACT.md`](ABSTRACT.md).

## Problem statement and approach

Tool-calling agents break when the API under them moves. The standard response
is to fine-tune on corrective traces; the standard failure is catastrophic
forgetting. HealGuard tests a stricter claim: a heal should deploy only if a
pre-registered gate shows it is better on the failure, no worse elsewhere, and
no less efficient.

```text
seeded world -> drift -> rollout -> deterministic classifier
  -> replay-verified pairs -> LoRA/DPO/KL-DPO -> HealGate
  -> DEPLOY / REJECT / NEED_MORE_DATA -> decision_log.jsonl
```

## Why this problem / real use case

Cloud audit agents are long-running, tool-heavy, and exposed to third-party API
changes they do not control. A pinned client that silently misses a bucket is
worse than one that errors: the audit returns clean while the account is exposed.
The practical question is not "can we fine-tune a fix?" but "can we prove the fix
did not break something else?" A 75% baseline and an 85% candidate will ship the
candidate; if it has an 11.7% silent failure rate, the team ships a regression
with a better headline.

## Research questions and pre-registered hypotheses

| Question | Result |
|---|---|
| Does raw success overstate competence? | **Yes:** 75/35 D1; 56.5/36.5 D5 |
| Can a runtime repair raise drift without changing clean? | **Yes:** 85/58.3, clean 73.3 |
| Does a +10 gain hide a new failure mode? | **Yes:** 11.7% residual + 3.3% premature |
| Does a harsher error make v0 fail? | **No:** D1-H 16/20 both, same failures |

Thresholds were frozen on 11 Nov, before the final D5 block: `min_n=40`; gain ≥
+0.10; one-sided paired bootstrap α=0.05, 10,000 iterations, seed 42; worst-family
drop ≤ 0.05; median steps ≤ 1.05×; redundant rate not higher; no new failure
label above 2%.

## Threat model [T3]

The attacker is the API change, not a human adversary. The agent has a pinned
registry, no out-of-band API-version signal, a 24-turn budget, read-only tools,
and a deterministic oracle. The gate protects against **silent capability
regressions**: a candidate that improves the target drift but degrades clean
traffic, efficiency, a family, or the failure taxonomy.

## Related work

LoRA (Hu et al., 2021), DPO (Rafailov et al., 2023), GRPO (Shao et al., 2024),
TRL `SFTTrainer`/`DPOTrainer`, AuditKIT 1.1.3, AgentTune `d4db4c1`, paired
bootstrap (Efron). The novel framing is the heal as a **release gate** with
family-stratified, grounded-efficiency, and new-failure-mode checks.

## Hackathon repos used

| Kit | Used? | How |
|---|---|---|
| **AgentTune** | Yes, limited | S1 CPU spine smoke; `agenttune.agentic`/`TrlAgenticDPO` evaluated. Final KL-DPO uses TRL directly because the custom tool-call template is not rendered by the stock path. |
| **AuditKIT** | Yes | `ak.scorer`, `ak.Score`, `ak.Direction`, `ak.AgentLoopDetection`, `ak.Sample`; six declared-direction scorers + loop detection. Gate is ours. |
| **CuratorKIT** | No | Synthetic world; no external corpus to filter. |
| **AlignTune** | No | DPO/KL-DPO via TRL. |
| **SafeTune** | No | Capability healing, not safety alignment. |
| **CircuitKIT** | No | LoRA adapter, not a circuit intervention. |

AgentTune is in the pin table and the S1 smoke; it is **not** the trainer for the
final adapters.

## How the repos connect

```text
AgentTune (S1 smoke, pinned)
        |
        v
AuditKIT scorers + loop detection ---> eval/rollout.py ---> gate-ready records
        ^                                                   |
        |                                                   v
world/oracle/expert/classifier  <--- trajectories --->  HealGate
        |                                                   |
        +--> replay-verified pairs --> TRL SFT/DPO/KL-DPO --+
                                                            |
                                                   decision_log.jsonl
```

The gate never calls an LLM. The only judge is the deterministic oracle plus the
rule classifier; AuditKIT supplies metric plumbing, not a verdict.

## Models and licenses

- **Base:** `CohereLabs/tiny-aya-global`, 3.35B, 8k, CC-BY-NC-4.0.
- **Quant:** 4-bit nf4, double quant; fp16/bf16 compute.
- **Adapters:** v0, v1_good, v1_sft, v2/v4/v5 run-log variants, v6 KL-DPO;
  inherit the base model's non-commercial research terms.
- **Teacher:** none. The expert is a deterministic Python policy, not a model.
- **Closed-model baseline:** none.
- **AuditKIT:** Lexsi Labs Source Available License 1.2.
- **AgentTune:** pinned `d4db4c12d52342227c4113a70183d0f64aa2f345`.

## Datasets, benchmarks, and licenses

All data is synthetic from `world/snapshot.py`; no public corpus, no scraped
data, no third-party labels. Dataset card: `bononi/healguard-data`.

| Split | Seeds | Size | Purpose |
|---|---|---:|---|
| train_clean | 0–399 | 400 | v0 SFT |
| dev | 3000–3039 | 40 | selection |
| drift_targeted | 6000–6059 | 60 | D1 gate |
| gate_regression | 10000–10059 | 60 | clean + D2/D3/D4 |
| fresh D1 | 40000–40059 | 60 | D1/guard comparisons |
| final D5 drift | 70000–70199 | 200 | final baseline |
| final D5 clean | 70200–70399 | 200 | regression bucket |
| D5 training | 71000–71199 | 200 | pairs only |
| sealed | 50 | 50 | hash-frozen; not opened |

**Languages:** English tool names/prompts; no multilingual claim. **Leak check:**
seed ledger reserves every evaluation split; the pair loader drops reserved
seeds; a test asserts no overlap; final D5 pairs checked to `71000–71199`. Direct
id matching is exact because the data is synthetic; no embedding/MinHash check is
needed.

**Benchmarks:** no public safety/general benchmark is the focus. AuditKIT's six
scorers are used for declared directions. The benchmark is the deterministic
world + oracle.

**Tool/task environments:** eight tools, three families, 2–4 buckets, 2–3 roles,
2–3 groups, answers 0–8 ids, 50% `none`, strict specs; full trajectories are
re-executable with `heal/replay.py`.

## Metrics and why

| Metric | Direction | Why |
|---|---|---|
| Raw task success | higher | Headline; leaderboard-comparable. |
| Grounded success | higher | Correct + every required evidence call; exposes the none floor. |
| Balanced success | higher | Mean of none/vulnerable; diagnoses the `[]` prior. |
| Success on vulnerable | higher | The task the project exists to solve. |
| Median successful steps | lower | Efficiency; more turns is a regression. |
| Redundant-call rate | lower | Loop/cost signal. |
| Failure-label share | bounded | New failure >2% blocks deploy. |
| Gain / bootstrap | higher | Paired effect and significance. |
| Family drop | lower | A mean cannot hide a family collapse. |
| Latency / tokens | lower | Serving cost; T4 measured. |

**Automatic judge:** no LLM. Oracle exact; classifier fixed. Battery 112/112,
macro-F1 1.000; field rollouts accepted-label 0.500, residual 0.000. Significance
is paired bootstrap, not a human rating. **Agentic metrics:** task success,
grounded success, steps, tool-call validity (replay), redundant calls, wall-clock
latency. Efficiency is compared only on tasks both runs solved with full
evidence.

## Method / experiments

| Item | Value |
|---|---|
| Base | `CohereLabs/tiny-aya-global`, 3.35B |
| Quant | 4-bit nf4, double quant |
| LoRA | r=16, α=32, dropout 0.05, all-linear, bias none |
| Trainable | 30.2M (0.89%) |
| v0 SFT | 400 traces, 5 epochs, 250 steps, LR 1e-4, assistant-only loss |
| DPO | LR 5e-5, 1 epoch, max length 2048 |
| KL-DPO | β=0.1, frozen v0 reference, 1 epoch, LR 5e-5 |
| v1_sft | 50/50 clean/corrective, 1 epoch, LR 5e-5 |
| Hardware | Colab T4 16GB; 5.13GB peak; 6.1s/step |
| T4 patches | `use_vllm=False`; fp16; cast bf16 trainables to fp32; gradient checkpointing |

Ablations: prompt-only control (delta 0.000); D1-H generic error (16/20 both,
null); 50/50 v2, higher-LR v4, 75/25 v5; guard; KL-DPO. Seeds are in the split
table; the final D5 block was untouched by training; the sealed set was not
opened.

## Results

D1, 60 tasks (95% Wilson intervals in [`ARTICLE.md`](ARTICLE.md) §12):

| Run | Raw drift | Grounded drift | Clean |
|---|---:|---:|---:|
| v0 | 45/60 = **75.0%** | 21/60 = **35.0%** | 44/60 = **73.3%** |
| v0 + Guard | 51/60 = **85.0%** | 35/60 = **58.3%** | 44/60 = **73.3%** |

Final D5, 200 tasks:

| Split | Raw | Grounded | None | Vulnerable |
|---|---:|---:|---:|---:|
| Drift | **56.5%** | **36.5%** | 82.7% (62/75) | 40.8% (51/125) |
| Clean | **80.0%** | **45.0%** | 97.7% (85/87) | 66.4% (75/113) |

Artifacts: `artifacts/runs/v0_d5_clean_D5_stale_drift_targeted.jsonl`,
`artifacts/runs/v0_d5_clean_clean_gate_regression.jsonl`, and the matching
`artifacts/trajectories/` files. v5/v6 candidate rollout files are not committed.

Candidate table; `†` = run log, not committed; `—` = not measured/not committed:

| Candidate | Drift raw | Drift grounded | Clean | Gate | Why |
|---|---:|---:|---:|---|---|
| v0 baseline | 75.0 | 35.0 | 73.3 | — | baseline |
| v1_good (DPO) | 73.3 | — | 75.0 | **REJECT** | no gain; steps 15 vs 12; redundant rise |
| v1_sft_a (SFT) | 83.3 | — | 61.7 | **REJECT** | +8.3 < +10; 23/60 schema (38%) |
| v1_sft_b (clean-mix) | — | — | — | **REJECT** | not committed |
| v2_recovery † | 78.0 | — | 72.0 | **REJECT** | clean regression |
| v4_recovery † | 80.0 | — | 63.0 | **REJECT** | clean collapse |
| v5_recovery † | 60.0 | — | 80.0 | **REJECT** | drift not fixed |
| v0 + Guard | 85.0 | 58.3 | 73.3 | **REJECT** | 11.7% residual + 3.3% premature |
| v1_poisoned | 63.3 | — | 71.7 | **REJECT** | corrupted canary |
| v6_recovery (KL-DPO) | — | — | — | **REJECT** | final D5; early-stop; files omitted |

**Findings:** the none floor; the guard side-effect; DPO verbosity (steps 15 vs
12); SFT forgetting (61.7 clean, 23/60 schema, F1 −16); D1-H null. **What didn't
work:** DPO minimal-edit, SFT corrective, clean-mix SFT, KL-DPO, guard. Sample
trajectory and full gate checks: [`ARTICLE.md`](ARTICLE.md) §12.

## Limitations

- N=60 for most comparisons; final D5 baseline N=200; v5/v6 candidate files not
  committed.
- Simulated cloud; three families, one template, one 3.35B model.
- D5 constructed last, accepts both names, expert-supervised; easier than D1.
- No sealed release because no candidate passed; opening it to rescue a failed
  candidate would contaminate the result.
- `†` rows are not artifact-backed.
- The guard is a runtime repair, not a fine-tuned model.
- English only; no multilingual/multi-template claim.

## With more credits

$200 HF credits. **GRPO with the deterministic oracle as reward:** k=4
trajectories/task, reward exact answer + evidence + steps − repeats/loops; LoRA
r=32, KL to v0. **Scale to 200 tasks/bucket:** pre-register `70000–70199` /
`70200–70399`, report 95% paired-bootstrap intervals. **Live ZeroGPU Space:** one
cached rollout through all checks; a refusal the judge can see. **D6/D7:** expiring
pagination and partial-failure APIs. **Gate calibration:** 100 synthetic
candidates, false-DEPLOY/false-REJECT. **Sealed release:** freeze, hash, run once,
publish the log including rejects. Forecast (not a result): ≥85% raw / ≥65%
grounded D5, clean ≥78%, no label above 2%.

## Artifacts

- HF article: https://huggingface.co/bononi/healguard-article
- Models: https://huggingface.co/collections/bononi/healguard-models
- Dataset: https://huggingface.co/datasets/bononi/healguard-data
- Space: https://huggingface.co/spaces/bononi/healguard-timeline
- GitHub: https://github.com/prarthana2031/HealGuard-Lexsi

## Documentation and reproduction

Python 3.12.10; `pip install -r requirements.txt`; no GPU for report/tests.
Training needs a T4 16GB or better.

```bash
python run_all.py --stage report     # artifacts/results.md
python -m pytest -q                  # GPU-free suite
python -m eval.rollout --help        # GPU harness
```

Repo layout: `world/` environment; `oracle/` truth; `expert/` teacher; `heal/`
drifts/replay/pairs; `classifier/` labels; `eval/` rollout; `gates/` HealGate;
`training/` recipes; `data/` splits; `artifacts/` results/logs. Leak check: seed
ledger + `tests/test_no_leakage.py`; D5 pairs checked to `71000–71199`.

## Declaration

No submitted component was produced before the hackathon window. The audit trail
starts 3 Nov 2026; all adapters, rollouts, and results were produced during the
event. Third-party code is limited to the pinned kits/libraries in
`requirements.txt` and `infra/commits.md`.

## Teammates and contributions

- **@prarthana2031** — world/oracle/expert, training infra, D5/guard integration,
  GPU runs.
- **Preethi HJ** — classifier, evaluation harness, gate, results, write-up.

Adjust handles to the final roster before submitting.

## LLM usage and changelog

DeepSeek V4.1 Flash was used for code, tests, and documentation; it did not design
the policy, reward, gate, or teacher. Roughly 60–70% of the prose was drafted with
AI assistance and checked against `artifacts/`. Every number is artifact-backed or
marked `†`. **v1 (final):** restructured to the official write-up format; no
experimental result changed.
