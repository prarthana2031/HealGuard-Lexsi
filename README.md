
```markdown
# 🛡️ HealGuard: A Verifiable Deployment Gate for Self-Healing Agents

**Track 3: Agentic Fine-Tuning (AgentTune + AuditKIT)**

## 📖 The Pitch
When a cloud API updates, it silently breaks the AI agents that rely on it. HealGuard is a fully automated, closed-loop pipeline that detects the failure, generates corrective training data, retrains the agent, and uses a **rigorous statistical gate** to decide if the new model is actually safe to deploy.

We built a deterministic, seeded fake-AWS environment and tested 8 healing strategies (DPO, SFT, KL-DPO, Runtime Guard). **Our Gate rejected all of them.** This repository is the proof that in the era of self-healing agents, a strict deployment gate is more valuable than a retrained model.

## 🔗 Submission Links
- **Hugging Face Article:** [Read the full research paper here](https://huggingface.co/bononi/healguard-article)
- **Model Collection:** [HealGuard Models](https://huggingface.co/collections/bononi/healguard-models)
- **Dataset Collection:** [HealGuard Data](https://huggingface.co/collections/bononi/dataset-collection)
- **Live Timeline Demo:** [Gradio Space](https://huggingface.co/spaces/bononi/healguard-timeline)
- **Dataset Repository:** [bononi/healguard-data](https://huggingface.co/datasets/bononi/healguard-data)

---

## 🏗️ Repository Structure

```text
heal-guard/
├── agent/
│   ├── schema.py            # Pydantic Trajectory models
│   └── template.py          # Custom Tiny Aya chat template with {% generation %}
├── artifacts/
│   ├── runs/                # Cached .jsonl model evaluations
│   ├── decision_log.jsonl   # Append-only Gate decisions
│   └── results.md           # Generated final tables
├── classifier/
│   ├── rules.py             # 11-label failure taxonomy (drift_mismatch, residual, etc.)
│   └── test_classifier.py   # 112/112 hand-verified tests
├── configs/
│   └── thresholds.yaml      # Frozen Gate thresholds (min_n=40, gain=+10%, etc.)
├── data/
│   ├── train_clean/         # 400 expert traces for SFT
│   ├── dev/                 # 40 paired tasks for early stopping
│   ├── drift_targeted/      # 60 tasks under D1 drift
│   └── final_sealed/        # 50 hashed, unopened tasks
├── eval/
│   ├── rollout.py           # Generation harness (HF/vLLM)
│   ├── auditkit_runner.py   # 7 metric scorers via AuditKIT
│   └── trajectory_adapter.py# JSONL -> AuditKIT format
├── expert/
│   └── scripted.py          # 100% Oracle expert (deterministic)
├── gates/
│   ├── deployment_gate.py   # Paired bootstrap, family regression, efficiency
│   └── registry.py          # Adapter versioning and rollback
├── heal/
│   ├── drifts/              # D1 (rename), D2 (paginate), D3, D4, D5 (silent)
│   ├── replay.py            # Replay verification (100% reward=1)
│   └── examples.py          # DPO pair generation with minimal edit
├── scripts/
│   └── run_all.py           # End-to-end orchestrator (CPU reproducible)
├── training/
│   ├── train_v0.py          # Base SFT (QLoRA 4-bit, r=16)
│   ├── train_kl_dpo.py      # KL-regularized DPO (beta=0.1)
│   └── train_heal_sft.py    # Catastrophic forgetting baseline
└── world/
    ├── snapshot.py          # Deterministic fake-AWS generator
    └── tools.py             # 8 tool APIs with strict schemas
```

---

## 🔬 Technical Deep Dive

### 1. The Environment (World v2)
We built a deterministic, seeded fake-AWS cloud. Each snapshot contains buckets, roles, and security groups. The Oracle uses 30 hand-verified rules (e.g., explicit `Deny` overrides `Allow`, `PublicAccessBlock` overrides ACLs).

**The "None" Floor:**
By design, 50% of tasks have an empty answer (`[]`). This prevents the model from learning a "find a vulnerability" prior and forces it to actually audit the state. The world generator ensures that every seed produces a unique, complex topology of IAM policies and S3 buckets.

### 2. The Drifts (D1-D5)
- **D1 (Rename):** `bucket` → `bucket_name`. The error message explicitly says "use bucket_name".
- **D2 (Paginate):** `list_buckets` returns 1 item per page.
- **D3 (Required):** Adds `region` as a required argument.
- **D4 (Tool Rename):** `get_bucket_acl` becomes `get_acl`.
- **D5 (Silent Empty):** The wrong argument returns `{}` instead of an error. This breaks the self-correction loop because the model doesn't know it failed.

### 3. The Gate (HealGate)
The Gate evaluates a candidate against `v0` using paired task IDs. A `DEPLOY` requires passing *all* of these:
- **Gain:** `delta >= 0.10` (10 percentage points).
- **Significance:** Paired bootstrap lower bound > 0 (alpha=0.05, N=10,000).
- **Regression:** No family (F1/F2/F4) drops by more than 5%.
- **Efficiency:** Median steps ≤ 1.05x baseline.
- **Safety:** No new failure mode > 2% (e.g., `premature_finish`, `residual`).

```python
def evaluate_deployment_gate(baseline, candidate):
    # Integrity check
    if not verify_integrity(baseline) or not verify_integrity(candidate):
        return {"decision": "NEED_MORE_DATA"}
    
    # Paired Bootstrap
    delta, lower_bound = paired_bootstrap(baseline, candidate)
    if delta < 0.10 or lower_bound <= 0:
        return {"decision": "REJECT", "reason": "gain"}
    
    # Family Regression
    if worst_family_drop(baseline, candidate) > 0.05:
        return {"decision": "REJECT", "reason": "regression"}
    
    # New Failure Modes
    if new_failure_share(candidate) > 0.02:
        return {"decision": "REJECT", "reason": "new_mode"}
        
    return {"decision": "DEPLOY"}
```

---

## 📊 The Candidate Table (All Rejected)

| Candidate | Intervention | Raw Drift / Clean | Decisive Failure |
|---|---|---:|---|
| **v0** | SFT Baseline | 75.0% / 73.3% | *Baseline* |
| `v1_good` | DPO (verified pairs) | 68.3% / 73.3% | Gain -6.7, steps (15 vs 12), redundant calls |
| `v1_sft_a` | SFT 23 traces | 0.0% / 0.0% | Output collapse (gibberish) |
| `v1_sft_b` | SFT 40+26 mix | 83.3% / 61.7% | Gain +8.3, clean F1 drop, 38.3% schema violations |
| `v2_recovery` | SFT 50/50 | 78.3% / 71.7% | Gain +3.3, clean F1 drop |
| `v4_recovery` | SFT high LR | 80.0% / 63.3% | Gain +5.0, clean F1 drop |
| `v5_recovery` | SFT 75/25 | 60.0% / 80.0% | Gain -15.0 |
| **`v0_guard`** | **Runtime Guard** | **85.0% / 73.3%** | **New failure modes (11.7% vs 2% limit)** |
| `v1_poisoned` | DPO Control | 63.3% / 75.0% | Gain -11.7 |
| `v6_recovery` | KL-DPO (D5) | 40.0% (15 tasks) | Early stopped (incomplete) |

---

## ⚙️ Technical Specifications

| Setting | Reported Configuration |
|---|---|
| Base Model | `CohereLabs/tiny-aya-global` (3.35B, 8k context, CC-BY-NC-4.0) |
| Quantization | 4-bit NF4 QLoRA (double quantization for KL-DPO) |
| LoRA Config | `r=16`, `alpha=32`, `dropout=0.05`, `target_modules="all-linear"` |
| v0 SFT | 400 traces; 20% recovery-trace injection; 5 epochs; LR `1e-4`; batch 1; GradAccum 8; max len 2048 |
| KL-DPO | `beta=0.1`; LR `5e-5`; 1 epoch; frozen v0 adapter as explicit reference |
| T4 Optimization | Custom `bf16 -> fp32` cast for the GradScaler to prevent Kaggle T4 crashes |
| Gate Bootstrap | 10,000 iterations, α=0.05, paired task IDs |

---

## 🚀 Roadmap (Round 2 Potential)
If shortlisted, we have a clear roadmap to push this pipeline to a **90+ score**:
1. **GRPO Implementation:** Use the $200 credits to run a 3-seed GRPO (Generative Reward Policy Optimization) stage on the `v6` adapter. Using the deterministic Oracle as the live reward function will allow the model to learn without static DPO pairs.
2. **Scale to 200 Tasks:** Increase the evaluation block from N=60 to N=200 paired tasks to establish statistical significance at a 95% confidence level.
3. **Live ZeroGPU Demo:** Upgrade the Gradio Space to a live endpoint, allowing judges to interact with the `v0_guard` model in real-time.
4. **Harsher Drifts:** Introduce D6 (silent argument rename) and D7 (tool deletion) to test the boundary of the Runtime Guard.
5. **Ablation Studies:** Run a full ablation on the KL-DPO beta parameter (0.01, 0.1, 0.5) to find the optimal anchor strength.

## 🏃 Reproduce
```bash
git clone https://github.com/prarthana2031/HealGuard-Lexsi.git
cd HealGuard-Lexsi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python run_all.py --stage all
```

## 📜 License
Tiny Aya is CC-BY-NC-4.0. All derivative adapters and code are for research use only.
```
