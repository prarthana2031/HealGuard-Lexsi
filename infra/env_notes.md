# Block 1 toolchain note

The one-page note Block 1 was supposed to leave behind. Everything here was
measured, not assumed.

## Pins

| Component | Pin | How it was verified |
|---|---|---|
| Python | 3.12.10 | `python --version` |
| AgentTune | `d4db4c12d52342227c4113a70183d0f64aa2f345` | `git -C ../AgentTune log -1` matches this file |
| AuditKIT | `71316da345784ff220d5ed59337441a59ab31526` | `git -C ../AuditKIT log -1` matches this file |
| transformers | 5.22.1 | `pip show transformers` |
| PEFT | 0.21.2 | `pip show peft` |
| TRL | 1.7.1 | `pip show trl` |
| bitsandbytes | installed | 4-bit nf4 path used by every training script |

Both libraries are installed from a local checkout at the pinned commit, not
from a floating `main`.

## S2 — the hard gate

| Question | Answer |
|---|---|
| Does the stock Cohere template emit a tool call? | **No.** It renders no tool block. |
| Does the custom `<tool_call>` template emit one? | **Yes**, parseable, in 6.1 s on a T4. |
| Peak GPU memory | 5.13 GB |
| Trainable parameters | 30.2 M (0.89 %) |
| Platform | Colab T4 16 GB |

`use_vllm=False` everywhere. vLLM is Linux+CUDA only and is how a free session
dies.

## The template we train with

Kept verbatim in `training/train_dpo.py:TOOL_TEMPLATE` and copied in
`training/template_notes.md`. Tools go in the system turn; tool results come back
as a `SYSTEM_TOKEN` turn prefixed with `Tool result:`.

## S3–S5

- **S3** `episodes_from_agenttune` exists; the precomputed path is what we use,
  because rollouts happen in our harness, not inside AuditKIT.
- **S4** `FullClosedLoop` moves with stub callables. We call the stage classes
  directly; the stock loop is a baseline, not our architecture.
- **S5** `FailureDetector.scan_audit_log` requires `trajectory_id`, `stage_name`
  and a `state_snapshot` dict on every line. We do not feed it a custom log; our
  classifier reads the trajectory directly. Written workaround, not a silent
  failure.

## GPU-free checks

`python -m pytest` runs 142 tests with no GPU, no network and no model download.
The full pipeline (`python run_all.py --stage all`) runs in under a minute on CPU.
