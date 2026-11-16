# HealGuard: A Verifiable Deployment Gate for Self-Healing Tool-Calling Agents

**Track 3 — Agentic Fine-Tuning** · Lexsi × Cohere Labs × Hugging Face

[Article](https://huggingface.co/bononi/healguard-article) ·
[Models](https://huggingface.co/collections/bononi/healguard-models) ·
[Dataset](https://huggingface.co/datasets/bononi/healguard-data) ·
[Space](https://huggingface.co/spaces/bononi/healguard-timeline) ·
[Code](https://github.com/prarthana2031/HealGuard-Lexsi)

> A cloud API changes under an agent. HealGuard detects the failure, attempts a
> heal, and deploys it only if a pre-registered statistical gate proves it is
> better on the failure and no worse anywhere else. Ten candidate entries, zero
> unsafe deploys.

---

## TL;DR

- **v0 is not 75% good; it is 35% grounded.** Half the tasks are `none`, so `[]`
  is free. Raw success was hiding the failure the project exists to fix.
- **A +10-point fix was rejected.** The Runtime Guard reached **85% raw /
  58.3% grounded** on D1 and left clean at **73.3%**, but introduced **11.7%
  residual** and **3.3% premature_finish** failures. The gate refused the trade.
- **No candidate passed.** DPO, SFT, clean-mix SFT, KL-DPO, pagination, and the
  guard were all blocked. The gate is the deliverable; the article is the
  post-mortem.

## Findings

### 1. The none-floor illusion

| Run | N | Raw success | Grounded success |
|---|---:|---:|---:|
| v0, D1 fresh block | 60 | **75.0%** | **35.0%** |
| v0, D5 final block | 200 | **56.5%** | **36.5%** |
| v0, clean final block | 200 | **80.0%** | **45.0%** |

Grounded success = correct answer **and** every required evidence call. The
50% `none` split means a model that gives up with `[]` clears the raw floor. The
gap between the two columns is the project's first finding.

### 2. The Runtime Guard was rejected

| Run | Raw drift | Grounded drift | Clean | Gate |
|---|---:|---:|---:|---|
| v0 | 75.0% | 35.0% | 73.3% | — |
| v0 + Guard | **85.0%** | **58.3%** | **73.3%** | **REJECT** |

The guard passed gain (+0.10), paired bootstrap (lower bound 0.0167), family,
efficiency, and the clean bucket. It failed **new_failure_mode_share**:
`residual` 7/60 = **11.7%**, `premature_finish` 2/60 = **3.3%**. A success-only
dashboard would have shipped a model that traded schema errors for silent wrong
answers.

## Repository map

```text
HealGuard-Lexsi/
├── world/            deterministic fake-AWS account, strict tool specs
├── oracle/           exact vulnerable-id set, no LLM judge
├── expert/           scripted teacher; 100% correct, recovery traces
├── heal/
│   ├── drifts/       D1 rename, D2 pagination, D3 region, D4 tool rename,
│   │                 D5 fail-closed empty payload
│   ├── replay.py     re-executes every tool call, recomputes reward
│   └── examples.py   replay-verified DPO pairs; D5 cue prefix
├── classifier/       11 rule labels; drift inferred from the trace
├── eval/             rollout harness, AuditKIT scorers, sealed runner
├── gates/            HealGate, decision log, adapter registry, rollback
├── training/         v0 SFT, DPO, SFT heal, KL-DPO, answer head
├── data/             splits, sealed set, heal pairs, drift-train rollouts
├── artifacts/        results.md, decision_log.jsonl, runs, trajectories
├── tests/            leakage, determinism, answer balance
├── configs/          thresholds.yaml, frozen 11 Nov
├── FINAL_RESULTS.md  final 200-task D5 table + gate decisions
├── FINAL_DAY_PLAN.md one-shot protocol, seed ranges, kill criteria
└── run_all.py        drift → detect → classify → generate → gate → report
```

## Technical deep dive

### World v2 — a deterministic cloud account

- Three independent families: S3 buckets, IAM roles, security groups.
- 2–4 buckets, 2–3 roles, 2–3 security groups per seed; answers range 0–8 ids.
- Every split is exactly 50% `none`; positions and counts are seed-derived.
- Oracle rules: public ACL/policy, explicit deny, public-access block, wildcard
  role action/resource, world-open ingress on 22/3389/3306.
- `CloudEnv` is pure Python; no network, no clock, no randomness at call time.
- Every tool returns a dict; list payloads capped at 25 items / 800 chars.

### Drifts D1–D5

| Id | Contract change | Intended failure |
|---|---|---|
| D1 | `bucket` → `bucket_name` on three tools | stale registry; helpful error |
| D1-H | same, but generic `invalid arguments` | tested; v0 guessed the obvious name |
| D2 | `list_buckets` caps at one bucket per page | caller ignores `next_token` |
| D3 | `region` becomes required | caller omits the new field |
| D4 | `get_bucket_acl` → `get_acl` | unknown tool / migration |
| D5 | `bucket` returns `{}`; `bucket_name` returns data | silent fail-closed miss |

D5 is a constructed, expert-supervised drift built last. It accepts both names
and is easier than D1; do not compare its raw number to D1 without that caveat.

### HealGate — five pre-registered checks

1. **Gain ≥ +0.10 absolute** on the targeted bucket.
2. **Paired bootstrap p < 0.05** (one-sided, 10,000 iterations, seed 42).
3. **No family regression > 5 points**; min 10 tasks per family.
4. **Efficiency ≤ 1.05×** the baseline, median successful steps on grounded
   successes; redundant-call rate must not rise.
5. **No new failure label above 2%** of the candidate bucket.

Plus: paired task coverage and `min_n = 40` per bucket. Outcomes are exactly
`DEPLOY`, `REJECT` (named failing checks), or `NEED_MORE_DATA`. Every decision
is appended to `artifacts/decision_log.jsonl` with every number and threshold.

### Training recipe

| Item | Value |
|---|---|
| Base model | `CohereLabs/tiny-aya-global`, 3.35B, 8k, CC-BY-NC-4.0 |
| Quantization | 4-bit nf4, double quant, fp16/bf16 compute |
| LoRA | r=16, α=32, dropout 0.05, all-linear, bias none |
| Trainable params | 30.2M (0.89%) |
| v0 SFT | 400 traces, 5 epochs, 250 steps, LR 1e-4, assistant-only loss |
| DPO heal | LR 5e-5, 1 epoch, max length 2048 |
| KL-DPO | β=0.1, frozen v0 reference, 1 epoch, LR 5e-5 |
| v1_sft | 50/50 clean/corrective, 1 epoch, LR 5e-5 |
| T4 patches | `use_vllm=False`; fp16; cast bf16 trainables to fp32; gradient checkpointing |
| Measured | 5.13 GB peak, 6.1 s/step, 30.2M trainable |

## Candidate results

Values are the 60-task D1 block unless noted. `†` = from the run log, not
committed; `—` = not measured / not committed. The final 200-task D5 table is in
[`FINAL_RESULTS.md`](FINAL_RESULTS.md).

| Candidate | Drift raw | Drift grounded | Clean raw | Gate | Why |
|---|---:|---:|---:|---|---|
| v0 baseline | 75.0 | 35.0 | 73.3 | — | baseline |
| v1_good (DPO) | 73.3 | — | 75.0 | **REJECT** | no gain; steps/redundant |
| v1_sft_a (SFT) | 83.3 | — | 61.7 | **REJECT** | +8.3 < +10; 38% schema |
| v1_sft_b (clean-mix) | — | — | — | **REJECT** | not committed |
| v2_recovery † | 78.0 | — | 72.0 | **REJECT** | clean regression |
| v4_recovery † | 80.0 | — | 63.0 | **REJECT** | clean collapse |
| v5_recovery † | 60.0 | — | 80.0 | **REJECT** | drift not fixed |
| v0 + Guard | 85.0 | 58.3 | 73.3 | **REJECT** | 11.7% residual + 3.3% premature |
| v1_poisoned | 63.3 | — | 71.7 | **REJECT** | corrupted canary |
| v6_recovery (KL-DPO) | — | — | — | **REJECT** | final-day D5; files omitted |

The artifact-backed rows are in
[`artifacts/results.md`](artifacts/results.md) and
[`artifacts/decision_log.jsonl`](artifacts/decision_log.jsonl). The uncommitted
rows are reported honestly rather than back-filled.

## Reproduce

CPU only, no model download:

```bash
python -m pip install -r requirements.txt
python run_all.py --stage report     # regenerates artifacts/results.md
python -m pytest -q                  # GPU-free suite
```

The final 200-task D5 block and the one-shot protocol are in
[`FINAL_DAY_PLAN.md`](FINAL_DAY_PLAN.md). The gate thresholds are frozen in
[`configs/thresholds.yaml`](configs/thresholds.yaml); do not edit them after the
sealed set opens.

## Round 2 roadmap

The next budget is $200 in HF credits. The plan is to turn the gate from a
post-mortem into a training loop.

1. **GRPO with the oracle as reward.** The oracle is deterministic and exact, so
   GRPO can sample k tool-call trajectories per task and reward exact answer
   sets, evidence coverage, and step efficiency. No human labels, no LLM judge.
   Start with k=4, 1 epoch, LoRA r=32; compare against the DPO/SFT Pareto front.
2. **Scale the gate to 200 tasks per bucket.** The current N=60 gives wide
   bootstrap intervals. N=200 (drift `70000–70199`, clean `70200–70399`) keeps
   the same effect size but tightens the confidence interval; pre-register the
   block before opening it.
3. **Live ZeroGPU demo.** A Gradio Space that runs one cached rollout through
   the gate, shows every check, and refuses deployment live. No GPU required for
   the judge; ZeroGPU for optional live generations.
4. **Harsher drifts D6/D7.** D6: stateful pagination with a token that expires
   silently. D7: a partial-failure API that returns one stale field alongside
   valid data. Both target failures that cannot be guessed from the error text.
5. **Calibrate the gate.** Report the false-DEPLOY rate on poisoned adapters and
   the false-REJECT rate on known-good policy records across 100 synthetic
   candidates. The gate should be judged like a classifier, not a vibe.
6. **Pre-registration and sealed release.** Freeze thresholds and splits, hash
   them, then run one sealed evaluation. Publish the decision log, not a
   hand-picked table.

Success in Round 2 is not "a higher number." It is a candidate that passes a
pre-registered gate on a held-out block, with a clean control and a reproducible
decision log.

## Limitations

- N=60 for most candidate comparisons; the final D5 baseline is N=200, but the
  candidate rollout files for v5/v6 are not committed.
- Simulated cloud account; three families, one chat template, one 3.35B model.
- D5 was built after the earlier nulls and accepts both argument names; it is an
  easier test than D1. Say so in any comparison.
- The Runtime Guard is a runtime repair, not a fine-tuned model.
- No commercial claim; research use only.

## Licenses

`CohereLabs/tiny-aya-global` is CC-BY-NC-4.0. AuditKit is under the Lexsi Labs
Source Available License. AgentTune is used at the pinned commit in
[`infra/commits.md`](infra/commits.md). See the article for the full licence
section.
