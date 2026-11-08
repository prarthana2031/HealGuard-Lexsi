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
| Block 3 — drift damage ≥ 20 pts | Measured | **Never measured** | Measured: 30 / 30 / 45 / 30 pts per drift |
| Block 3 — classifier F1 ≥ 0.85 | Against injected faults | **0.0 on real failures**; "1.0" on self-authored mocks | 112/112 on a real battery; 0% residual |
| Week 2 — replay-to-1 examples | 100% of chosen | **Not built** | 30 pairs, 100% replay-verified |
| Week 2 — poisoned rejection | Gate blocks it | Gate could **never** return DEPLOY (N=20 < 40) | DEPLOY vs REJECT on real records |
| Week 2 — run_all | drift→detect→classify→generate→retrain→gate→report | `train` + `gate` only, wrong paths | All seven stages + report-only |
| World shape | 3 families, varied answers | One coin; answers were `[]` or 3 ids | **World v2**: independent families, 0–8 findings |
| Week 2 — article/collections/Space | Public | **Absent** | Still **absent** |
| Tests | Green | 112 | 149 |

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
| H9 | ~~**Broken `world/tools.py` semantics.** Calls after `finish` returned `env_error` instead of being ignored and flagged; tool output was never capped.~~ **FIXED.** Post-finish calls now return `{"ignored": true}` and increment `env.ignored_calls`, so they can no longer be mislabelled as injected environment errors. Every tool payload is capped (25 list items, 800-char strings) with a `truncated_items` counter. The D2 wrapper passes the ignore payload through instead of raising `KeyError`. | `world/tools.py`, `tests/test_world.py` | Fixed |

---

## 4. Medium findings

| # | Finding |
|---|---|
| M1 | ~~**The world is degenerate.**~~ **RESOLVED in World v2.** `snapshot.py` v1 flipped all three families together, so a vulnerable task always had exactly **3** ids at `…-0`. Sealed answer sizes were `{0: 27, 3: 23}`; `always-none` scored the none-rate; the per-family table measured one correlated variable. v2 draws families independently and positions uniformly: answer sizes `{0,1,2,3,4,5,6,8}`, 43% of vulnerable tasks are single-family, all-three is 1.7%, index distributions uniform. Drops now differ per drift (30/30/45/30) instead of being identical, which is the tell that it is fixed. | Measured over 4,000 seeds | Fixed |
| M2 | ~~**`always-none` is not near the floor.**~~ **RESOLVED.** Canaries are measured and published, and **balanced success** sits beside task success: `always-none` scores 0.500 task success, 0.500 balanced, **0.000 on vulnerable tasks**. Every split is now exactly 50/50 by construction, so the canary is pinned rather than incidental. | [`artifacts/results.md`](artifacts/results.md) | Fixed |
| M3 | **D2 deviates from the plan's literal wording.** Plan: "list return becomes `{items, next_token}` instead of a bare list". Code: clean is already `{items, next_token}`, D2 renames to `{bucket_list, continuation}`. Coherent (clean is paginated per §5) but the article must state the deviation, per the plan's own "say so" rule. |
| M4 | **One template only**, so the plan's "held-out templates" is vacuous. Recorded in `tests/test_no_leakage.py` rather than pretended. |
| M5 | **`compare.py`-style stock-loop comparison absent.** The plan allows a written paragraph instead; that paragraph is not written yet. |
| M6 | **`scripts/call_deepseek.py` sits in the repo root path with a live key** in `.env`. `.env` is correctly gitignored, but the key has been in a plaintext file next to the pipeline; rotate it and keep the script clearly marked as authoring-only. |
| M7 | ~~**`classifier/taxonomy.md` says "exactly one label"; the plan says primary + full set.**~~ **FIXED.** The taxonomy now states that the first match is the primary and every other match is kept, with the reason a drifted call is usually both `drift_mismatch` and `schema_violation`. | `classifier/taxonomy.md` | Fixed |
| M8 | ~~**Duplicate test** (`test_steps_on_success_uses_median` defined twice).~~ **FIXED.** Also removed: `_assistant_positions` dead code, a no-op branch in `run_all.py`, and the schema example that called a tool which does not exist (`check_bucket_acl`). | `eval/test_auditkit_runner.py`, `heal/examples.py`, `run_all.py`, `agent/trajectory_schema.md` | Fixed |

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

Verified now: `python -m pytest -q` → **149 passed**; `python run_all.py --stage all`
→ DEPLOY + REJECT, 30/30/45/30-point drift damage, 30 replay-verified pairs.
The sealed set was re-frozen once for world v2, before it was ever opened.

---

## 5a. World v2 (the second pass)

Requested after the first audit, executed the same day. What changed:

| File | Change |
|---|---|
| [`world/snapshot.py`](world/snapshot.py) | Rewritten: independent family draws, uniform positions, 2–4 buckets, decoys in every family, answer sets 0–8 |
| [`data/splits/build_splits.py`](data/splits/build_splits.py) | Seed ledger with exactly 50/50 `none` per split; reserved blocks; `gate_regression` keeps 42 clean + 6 D2 + 6 D3 + 6 D4 |
| [`data/splits/sealed_freeze.md`](data/splits/sealed_freeze.md) | Re-freeze record: both hashes, the reason, and the statement that the set was never opened |
| [`data/archive/world_v1/`](data/archive/world_v1/README.md) | The 80 real v1 rollouts, archived and labelled. They cannot be reused: under v2 the same seed is a different account |
| [`data/splits/seed_ledger.json`](data/splits/seed_ledger.json) | Every assigned seed, so a regeneration is auditable |
| [`gates/deployment_gate.py`](gates/deployment_gate.py) | Two real bugs found by the new data — see below |
| [`tests/test_no_leakage.py`](tests/test_no_leakage.py) | Asserts the v2 properties: exactly half `none`, ≥4 answer sizes, index 0 under 45% of findings, at least one single-family task |

**Two gate bugs that world v2 exposed.** The gate rejected its own ceiling, which
is how they surfaced:

1. `task_success_gain ≥ 0.10` was applied to **both** buckets. The plan gates the
   gain only on the bucket the fix targets; the regression bucket has to *not get
   worse*. A candidate that fixes the drift and leaves clean tasks unchanged is
   the normal, correct outcome, and it was being rejected.
2. `steps_on_success` compared two medians taken over **different success sets**.
   On a drifted task the broken baseline succeeds *because* it did less work — it
   could not read the listing, skipped every resource, and answered `[]`, which is
   correct for a `none` task. The healer was penalised for doing the reads it now
   can do. Efficiency is now paired, on the clean-API subset, with the unpaired
   medians still reported.

Both fixes are covered by new tests
([`gates/test_deployment_gate.py`](gates/test_deployment_gate.py)) and documented
in [`gates/README.md`](gates/README.md).

---

## 5b. The model path (third pass)

`eval/rollout.py` is the bridge between a finished adapter and the rest of the
pipeline. One command produces both halves:

```bash
python -m eval.rollout --adapter adapters/v0/final --bucket drift_targeted --tag v0
```

- **records** → `artifacts/runs/v0_drift_targeted.jsonl`, in exactly the contract
  the gate reads (`success`, `num_steps`, `total_tool_calls`, `redundant_calls`,
  `family_success`, `failure_mode`, `is_none_task`);
- **trajectories** → `artifacts/trajectories/`, or `data/drift_train/` with
  `--trajectories-dir`, which is where the healer looks.

`--prompt-only-control` runs the clean prompt and the D1 hint on the same seeds
and prints the delta, which is the "why not just update the prompt?" number. The
five prompt arms are one sentence each. `--compare-to artifacts/runs/v0` gates
the new run against a baseline and logs the decision.

**A bug this exposed, fixed the same hour.** `heal.examples.build_pair` required
the failure's message list to be *byte-identical* to the expert's up to the
divergence point. A real model never produces the expert's exact whitespace, so
it would have returned `None` for essentially every real rollout — the heal would
have silently produced zero pairs on the night. Pairs are now rebuilt from the
*canonical call sequence*, with each call re-executed for its real tool result,
so the shared prefix is guaranteed rather than hoped for. A prompt-hinted run is
still refused: it starts from a different system prompt and is a control, not
training data. Both behaviours are pinned by tests in
[`eval/test_rollout.py`](eval/test_rollout.py), including one that deliberately
corrupts the whitespace of a rollout and still expects a pair.

---

## 5c. Block 3 failure: v0 at 0.35 (8 Nov)

The first world-v2 v0 run scored **0.350** dev task success: `none` 0.650,
`vulnerable` 0.050, median 17.5 steps, **100% of tasks called a tool**, labels
`premature_finish` ×12 and `residual` ×14. No `schema_violation`, no
`ungrounded_answer`, no loops.

That mix says the format transferred and the *decision rule* did not. Three
causes, found in this order:

**1. The loss mask was never verified, and could not have worked.**
`assistant_only_loss=True` does not read role names. TRL looks for
`{% generation %}` markers and, if absent, tries to patch the template by exact
string match against a hard-coded list of known templates
(`trl/chat_template_utils.py:952-1024`). Our custom template is not on that list,
so the run either raised or trained on every token. The token budget of one
expert trace is **24% system prompt, 30% tool results, 45% assistant**: an
unmasked run spends most of its gradient reproducing text it cannot predict and
does not need.

**2. There were 39 optimizer steps.** 100 traces / batch 1 / grad-accum 8 /
3 epochs = 13 × 3. That is enough to learn a constant format and nowhere near
enough to learn "PAB overrides ACL, deny overrides allow, wildcard needs both
fields, risky ports are 22/3389/3306".

**3. Model selection used `eval_loss`.** With an unmasked loss, the best-loss
checkpoint is the one that best predicts *tool results*, which is not the
checkpoint that answers correctly.

**4. Every training trace carried a 503.** The plan says "include a *few* traces
where a tool errors"; all 100 had one. The model learns to re-issue
`list_buckets` with `limit=5` even when the first call succeeded, which is a
plausible source of the 17.5 median against the expert's ~14.

Fixes shipped: one template in `training/template.py` with generation markers
(they emit no text — rendering is byte-identical, asserted in
`training/test_template.py`); a 30-second `python -m training.check_masking`
that aborts on a fraction near 0 or 1 and on a span without `<tool_call>`;
`train_v0.py` takes `--epochs`/`--lr`/`--max-steps`, runs the mask check before
training, drops loss-based early stopping and saves every 50 steps for selection
by task success; `training/config.yaml` records 400 traces × 5 epochs = 250 steps
and a 20% error-trace share.

---

## 6. What only you can do (critical path, 7 days)

In order. Nothing here is optional for a competitive packet.

1. **Train v0 on world v2** (`training/train_v0.py`, free T4, ~16 min). The data
   changed, so the v1 checkpoint learned the wrong recovery *and* a world that no
   longer exists. Then replace `artifacts/runs/*.jsonl` with model records. This
   is the number the whole packet rests on.
2. **Generate v2 drift rollouts** into `data/drift_train/` (see the README there)
   once v0 can call tools, then `python -m heal.examples --source rollouts`.
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

**D1 — redesign the world? → DONE (World v2).**
Families are decoupled, answer sizes vary from 0 to 8, positions are uniform, and
every split is exactly half `none` by seed selection. The sealed set was
re-frozen once, before it was ever opened; both hashes and the reason are in
[`data/splits/sealed_freeze.md`](data/splits/sealed_freeze.md). Effects measured:

| Property | v1 | v2 |
|---|---|---|
| Answer-set sizes | `{0, 3}` | `{0,1,2,3,4,5,6,8}` |
| Single-family vulnerable task | never | 43% |
| All three families at once | always | 1.7% |
| `always-none` task success | 0.60 | 0.50 |
| Drift drops | 40 / 40 / 40 / 40 | 30 / 30 / 45 / 30 |

The last row is the proof: v1 could not produce different drifts because one
missed bucket made the whole answer set wrong. v2 can, and does.

**D2 — add balanced accuracy beside task success? → DONE, keep it.**
`balanced_success = (success on none-tasks + success on vulnerable tasks) / 2` is
now computed for every condition and every canary, and published in
[`artifacts/results.md`](artifacts/results.md). The effect is immediate:

| Policy | Task success | Balanced | On `none` | On vulnerable |
|---|---:|---:|---:|---:|
| always-none | 0.500 | **0.500** | 1.000 | **0.000** |
| always-all | 0.000 | 0.000 | 0.000 | 0.000 |
| finish-immediately | 0.500 | **0.500** | 1.000 | **0.000** |
| repeat-one-call | 0.500 | **0.500** | 1.000 | **0.000** |

`always-none` used to look like it cleared the 50% v0 bar. It now obviously does
not. Report both columns in the article; use balanced success only as the
diagnostic, never as a replacement for the pre-registered task-success threshold.

**D3 — keep the sealed hash? (recommendation: keep)**
50 tasks is inside the plan's 40–60 window. Re-freezing buys nothing and creates a
git-visible break in a commitment. Keep it, and say "one template, 50 tasks" in
limitations.

---

## 8. The three sentences that matter for the article

1. **A drift that hurts:** each injected drift costs the baseline 30, 30, 45 and
   30 points on held-out seeds — measured, not asserted, and now *different per
   drift* because world v2 lets a drift break one family at a time.
2. **Examples that replay:** 30 minimal-edit pairs for D1, 100% of chosen traces
   replay to a perfect reward, asserted in CI on the emitted file, not the manifest.
3. **A gate that blocks a bad retrain:** the good candidate DEPLOYs on a one-sided
   paired bootstrap; the poisoned candidate is REJECTed on six named checks, each
   printed with its number.
