# HealGuard audit against the 15-page plan

**Audited:** 7 Nov 2026, 19:25–20:10 IST.
**Plan:** `HealGuard_Weekly_Plan (1).pdf` (15 pages, 2 Nov 2026).
**Repo:** `heal-guard` @ `87d205c`, working tree had only `README.md` modified.
**Verdict:** the *engineering* is better than the repo suggests; the *evidence* is
much worse than the artifacts claim. Several numbers in the old README were
unverifiable or false, and three defects would have cost the science outright.
All three are fixed. The remaining risk is not code — it is that no model
checkpoint exists.

---

## 1. Scorecard

| Area | Plan asks for | Was | Now |
|---|---|---|---|
| Block 1 — smokes, template, pins | S1–S5 + toolchain note | Pins real, note **missing** (0-byte files) | Note written, pins re-verified |
| Block 2 — world, oracle, expert | Expert 100%, random <5%, determinism, 30 hand cases, sealed hash | All present and green | Same, plus drift coverage |
| Block 2 — splits | No id in two splits, none-rate 40–60% | **Leaked seeds**; `drift_targeted` not drifted | Rebuilt, asserted |
| Block 3 — metrics | 7 scorers, explicit direction, direction regression test | 5 scorers, no direction test | 7 + direction test |
| Block 3 — untuned floor | Recorded | Recorded (0/20) | Same, now reproducible |
| Block 3 — v0 ≥ 50% | A number | **No number, no adapter** | Still **missing** (GPU) |
| Block 3 — drift damage ≥ 20 pts | Measured | **Never measured** | Measured: 40 pts each |
| Block 3 — classifier F1 ≥ 0.85 | Against injected faults | **0.0 on real failures**; "1.0" on self-authored mocks | 112/112 on a real battery; 0% residual |
| Week 2 — replay-to-1 examples | 100% of chosen | **Not built** | 40 pairs, 100% replay-verified |
| Week 2 — poisoned rejection | Gate blocks it | Gate could **never** return DEPLOY (N=20 < 40) | DEPLOY vs REJECT on real records |
| Week 2 — run_all | drift→detect→classify→generate→retrain→gate→report | `train` + `gate` only, wrong paths | All seven stages + report-only |
| Week 2 — article/collections/Space | Public | **Absent** | Still **absent** |
| Tests | Green | 112 | 142 |

---

## 2. Critical findings

### C1. The classifier returned `residual` for 100% of real failures

**Evidence.** Running the shipped classifier on the 80 real drift rollouts in
`data/drift_train/v0_failed_rollouts.jsonl`:

```
D1 {'residual': 20}   D2 {'residual': 20}
D3 {'residual': 20}   D4 {'residual': 20}
```

`drift_mismatch` — priority **#2**, the label the demo's two-minute script says
out loud — had no implementation at all. A comment said it "requires
environment/tool configuration evidence" and then never looked for any.

**Why it was hidden.** `classifier/confusion_matrix.md` reported macro-F1
**1.0000**. It was scored on `tests/fixtures/mock_trajectories_20.jsonl`, sixteen
JSON blobs written by the same hand that wrote the rules, with `expected_labels`
written to match. A judge who reads both files sees a tautology.

**Fixed.** `classifier/rules.py` rewritten:

- `drift_mismatch` is inferred **from the trace** — an environment error whose
  argument or tool name appears in the schema diff, or a response exposing the
  new field. The classifier is never told the drift id; a test deletes the
  `drift` field and asserts the verdict does not change.
- `schema_violation` now covers the most common real failure: **no parseable tool
  call at all**. That alone was 80/80 of the untuned rollouts.
- Every match is kept, not just the primary (`labels`), and
  `first_divergent_step` is reported so the healer builds one example per
  cascade, not five.
- Argument validation is driven by `world/specs.py`, not by hard-coded checks on
  one tool.

**New evidence.** A battery where each label is produced by running a broken
policy against the *real* environment, label fixed before classification:
**112/112, macro-F1 1.0000 over all 11 labels**, 0% residual
([`classifier/confusion_matrix.md`](classifier/confusion_matrix.md)). The honest
field number is also reported: 0.5 accepted-label share on the untuned rollouts,
because a model that emits prose has no drift signature to find. Both numbers are
in the artifact; the old fake one is gone.

### C2. The gold training set taught the exact behaviour the project fixes

**Evidence.** `expert/scripted.py` retried a 503 with the *identical* call:

```
calls: {"list_buckets", {}}   # 503
       {"list_buckets", {}}   # identical retry
```

Result: **100 of 100** `train_clean` traces contained a duplicated call
(1,700 calls, 100 duplicate pairs). The plan's own words: *"Include a few traces
where a tool errors and the next call changes. v0 never learns recovery from
traces that never recover."* The teacher was teaching `unrecovered_error` and
`redundant_call`.

**Fixed.** The retry now changes the call (`limit=5`), asserted in
`tests/test_world.py`. `train_clean` regenerated: 0 duplicates, 0 errors, all
100 traces still 100% correct. **v0 must be retrained** — a checkpoint trained on
the old traces learned the wrong recovery.

### C3. The gate could never say DEPLOY, and never saw a failure

Two independent blockers:

1. `min_n_per_bucket = 40`, but both gate buckets had **20** tasks. Every run
   returned `NEED_MORE_DATA`, forever.
2. `data/drift_targeted` — gate bucket 1, "did we fix the failure?" — contained
   **clean** expert traces. `generate_splits.py` said `# Will be wrapped later`
   and nothing ever wrapped them. The gate was comparing clean to clean.

**Fixed.** `data/splits/build_splits.py` replaces the old generator:
`drift_targeted` is now 60 D1-drifted tasks on held-out seeds; `gate_regression`
is 42 clean + 6 D2 + 6 D3 + 6 D4, so a D1 fix that breaks D2 cannot hide in a
good average. The gate now returns **DEPLOY** for the good candidate and
**REJECT** for the poisoned one with seven named failing checks.

### C4. The healer trained on the tasks the gate graded

**Evidence.** `drift_targeted` used seeds **2000–2019**.
`data/drift_train/v0_failed_rollouts.jsonl` holds real model failures on seeds
**2000–2019**. The plan's single hardest rule — *"Drift-targeted … never used to
build examples"* — was violated by the data, and
`tests/test_no_leakage.py` only compared against the sealed set, so it stayed
green.

**Fixed.** New seed ledger in `data/splits/build_splits.py`; `drift_targeted`
moved to 6000–6059, `gate_regression` to 6500–6559, sealed untouched at
5000–5049. The leakage test now checks every pair of splits *and* the real
rollouts, and asserts the sealed hash still matches.

---

## 3. High findings

| # | Finding | Evidence | Status |
|---|---|---|---|
| H1 | **No v0 checkpoint anywhere.** `adapters/` held three empty directories. README claimed "v0 SFT trained, eval loss 0.05693" with no artifact, log or hash behind it. | `Get-ChildItem adapters` → empty | **Open — needs GPU.** Pipeline no longer depends on it |
| H2 | **`assistant_only_loss=False`.** Loss included the system prompt and every tool result, so a low eval loss can coexist with a model that still cannot call a tool. | `training/train_v0.py:91` | Fixed → `True` |
| H3 | **`train_heal.py` pointed at a file that has never existed** (`data/drift_train/dpo_pairs.jsonl`). `v1_good` was unbuildable. `train_poisoned.py` did not use DPO at all, so the two adapters were not "the same recipe" and the rejection proved less than claimed. | file absent; `train_poisoned` is SFT | Fixed → shared `training/train_dpo.py` |
| H4 | **Two contradictory trajectory contracts.** `agent/trajectory_schema.md` says `id`/`success`/`tool_calls`; `agent/schema.py` says `task_id`/`is_correct`; the data used a third. Every consumer needed an adapter. | read both files | Fixed → one model mirroring the markdown |
| H5 | **`metrics.py` had 5 of 7 scorers.** `tool_selection_acc` and `arg_accuracy` were absent; the plan's explicit "worse redundant rate must not grade as an improvement" test was absent. | `METRICS` list | Fixed → 7 scorers + direction test |
| H6 | **Gate vocabulary and evidence.** Outcomes were `PASS/FAIL/BLOCKED`, not `DEPLOY/REJECT/NEED_MORE_DATA`; no decision log; no adapter registry, no rollback. | `gates/deployment_gate.py` | Fixed → both vocabularies, JSONL log, registry + rollback test |
| H7 | **`run_all.py` did not run the pipeline.** It ran three training scripts and a gate read from files that never existed. No detect/classify/generate/report, no report-only mode. | `scripts/run_all.py` | Fixed → root `run_all.py`, seven stages |
| H8 | **`artifacts/results.md` did not exist**, yet the README linked to it. The plan's "no number in the article is typed by hand" had no generator. | dir absent | Fixed → generated by `--stage report` |
| H9 | **Broken `world/tools.py` semantics.** Calls after `finish` returned `env_error` instead of being ignored and flagged; tool output was never capped. | read | **Open** (low risk, 30 min) |

---

## 4. Medium findings

| # | Finding |
|---|---|
| M1 | **The world is degenerate.** `snapshot.py` flips all three families together, so a vulnerable task always has exactly **3** ids, always `bucket-…-0`, `role-…-0`, `sg-…-0`. Answer-set sizes over the sealed set: `{0: 27, 3: 23}`. Consequences: `always-none` scores ≈ the none-rate (**54%** on sealed, **60%** on dev) which is above the v0 floor of 50%; a model can shortcut "check bucket 0"; and the per-family table the gate wants is measuring one correlated variable. **Not fixed** — see decision D1. |
| M2 | **`always-none` is not near the floor.** The plan wants canaries near the floor, but with a ~50% none-rate `always-none` *is* ~50%. It measured **0.600** task success against a v0 bar of 0.500. **Partly fixed** — all four canaries are now measured and published, and **balanced success** is reported beside task success: `always-none` scores 0.500 balanced with **0.000 on vulnerable tasks**, so the free ride is visible. See D2. |
| M3 | **D2 deviates from the plan's literal wording.** Plan: "list return becomes `{items, next_token}` instead of a bare list". Code: clean is already `{items, next_token}`, D2 renames to `{bucket_list, continuation}`. Coherent (clean is paginated per §5) but the article must state the deviation, per the plan's own "say so" rule. |
| M4 | **One template only**, so the plan's "held-out templates" is vacuous. Recorded in `tests/test_no_leakage.py` rather than pretended. |
| M5 | **`compare.py`-style stock-loop comparison absent.** The plan allows a written paragraph instead; that paragraph is not written yet. |
| M6 | **`scripts/call_deepseek.py` sits in the repo root path with a live key** in `.env`. `.env` is correctly gitignored, but the key has been in a plaintext file next to the pipeline; rotate it and keep the script clearly marked as authoring-only. |
| M7 | **`classifier/taxonomy.md` says "exactly one label"; the plan says primary + full set.** Now both are produced; the doc should be updated to match (one paragraph). |
| M8 | **Duplicate test** (`test_steps_on_success_uses_median` defined twice in `eval/test_auditkit_runner.py`). Harmless, sloppy. |

---

## 5. What I changed today

New files:

```
run_all.py                          seven-stage orchestrator + report mode
AUDIT.md                            this document
world/specs.py                      strict arg specs, one source of truth
heal/drifts/schema_diff.py          machine-readable D1–D4 diffs
heal/replay.py                      re-execute a trace, recompute the plan's reward
heal/examples.py                    minimal-edit DPO pairs, replay-verified
heal/test_examples.py               the "100% replay to 1" CI assertion
classifier/build_failure_battery.py real-env failure battery, labelled by construction
classifier/fixtures/failure_battery.jsonl   112 trajectories, 11 labels
data/splits/build_splits.py         seed ledger, drifted buckets, leakage assertion
data/splits/sealed.py               sealed set as a hash-checked task list
eval/run_agent.py                   stale / expert / poisoned policies → gate records
gates/registry.py                   decision log + adapter registry + rollback
gates/test_registry_and_decisions.py
training/train_dpo.py               shared DPO recipe for heal and poison
artifacts/results.md                generated
artifacts/decision_log.jsonl        generated
```

Rewritten: `classifier/rules.py`, `expert/scripted.py`, `classifier/evaluate_classifier.py`,
`classifier/test_classifier*.py`, `eval/metrics.py`, `eval/auditkit_runner.py`,
`data/splits/*`, `training/train_v0.py`, `train_heal.py`, `train_poisoned.py`,
`tests/test_no_leakage.py`, `README.md`, `world/README.md`, `expert/README.md`,
`training/config.yaml`, `training/template_notes.md`, `infra/*.md`, `agent/schema.py`.

Deleted: `tests/fixtures/mock_trajectories_20.jsonl`, `tests/fixtures/expected_labels.json`,
`tests/generate_mock_trajectories.py` — the self-fulfilling classifier fixture.

Regenerated: `train_clean` (no duplicate calls), `dev`, `drift_targeted`
(actually D1), `gate_regression` (clean + other drifts), `data/heal/*`.
**The sealed set was not touched**; its hash still matches the committed value.

Verified now: `python -m pytest -q` → **142 passed**; `python run_all.py --stage all`
→ DEPLOY + REJECT, 40-point drift damage, 40 replay-verified pairs.

---

## 6. What only you can do (critical path, 7 days)

In order. Nothing here is optional for a competitive packet.

1. **Train v0 with the fixed data** (`training/train_v0.py`, free T4, ~16 min).
   Then run `eval/run_agent.py`'s equivalent against the model and replace
   `artifacts/runs/*.jsonl`. This is the number the whole packet rests on.
2. **Decide D1** (world redesign) before you train, because it changes the data.
3. **Train v1_good and v1_poisoned** on `data/heal/*` (one GPU job at a time).
4. **Run the gate on model records**, not policy records, and commit the decision
   log. The log is the evidence; the article is a view of it.
5. **Open the sealed set once, on 12 Nov**, score untuned / v0 / prompt-only / v1_good /
   expert, run the four canaries, write `results.md` from the script only.
6. **Publish**: article (nine sections), two HF collections, CPU Space with the
   cached timeline. The Space must show the break *and* the refusal.
7. **Freeze 14 Nov 18:00 IST.** Fresh clone, regenerate the table, ship.

Still missing and cheap to add: the prompt-only D1 control (one sentence in the
system prompt, measure, write down what happens) and the stock-loop paragraph.

---

## 7. Decisions for you

**D1 — redesign the world? (my recommendation: yes, today)**
The degenerate generator (M1) is the one thing a sharp judge can use to dismiss
the *result* rather than the code. Fixing it means decoupling the three families,
varying answer-set size, and regenerating every split — including re-freezing the
sealed hash, which is legitimate only because the set has not been opened and the
hash has no public artifact pointing at it yet. Cost: half a day plus a v0 retrain.
Payoff: `always-none` drops toward 0, answers stop being 3 ids, and the per-family
table becomes real.

**D2 — add balanced accuracy beside task success? → DONE, keep it.**
`balanced_success = (success on none-tasks + success on vulnerable tasks) / 2` is
now computed for every condition and every canary, and published in
[`artifacts/results.md`](artifacts/results.md). The effect is immediate:

| Policy | Task success | Balanced | On `none` | On vulnerable |
|---|---:|---:|---:|---:|
| always-none | 0.600 | **0.500** | 1.000 | **0.000** |
| always-all | 0.000 | 0.000 | 0.000 | 0.000 |
| finish-immediately | 0.600 | **0.500** | 1.000 | **0.000** |
| repeat-one-call | 0.600 | **0.500** | 1.000 | **0.000** |

`always-none` used to look like it cleared the 50% v0 bar. It now obviously does
not. Report both columns in the article; use balanced success only as the
diagnostic, never as a replacement for the pre-registered task-success threshold.

**D3 — keep the sealed hash? (recommendation: keep)**
50 tasks is inside the plan's 40–60 window. Re-freezing buys nothing and creates a
git-visible break in a commitment. Keep it, and say "one template, 50 tasks" in
limitations.

---

## 8. The three sentences that matter for the article

1. **A drift that hurts:** each injected drift costs the baseline 40 points on
   held-out seeds, measured, not asserted.
2. **Examples that replay:** 40 minimal-edit pairs for D1, 100% of chosen traces
   replay to a perfect reward, asserted in CI on the emitted file, not the manifest.
3. **A gate that blocks a bad retrain:** the good candidate DEPLOYs on a one-sided
   paired bootstrap; the poisoned candidate is REJECTed on seven named checks,
   each printed with its number.
