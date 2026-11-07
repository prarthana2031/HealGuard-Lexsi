#!/usr/bin/env python3
"""HealGuard end-to-end orchestrator.

    python run_all.py --stage all      # drift -> detect -> classify -> generate -> gate -> report
    python run_all.py --stage drift
    python run_all.py --stage detect
    python run_all.py --stage classify
    python run_all.py --stage generate
    python run_all.py --stage retrain
    python run_all.py --stage gate
    python run_all.py --stage report   # regenerates the table from cached artifacts, no GPU

`report` and `gate` read cached records only, so Friday's fresh-clone check can
regenerate the main table without retraining anything.

Every stage writes into `artifacts/`. Nothing in this file needs a GPU; the
`retrain` stage shells out to the training scripts, which do.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

ARTIFACTS = ROOT / "artifacts"
RUNS = ARTIFACTS / "runs"
DECISION_LOG = ARTIFACTS / "decision_log.jsonl"

BUCKETS = {
    "drift_targeted": "data/drift_targeted/expert_traces.jsonl",
    "gate_regression": "data/gate_regression/expert_traces.jsonl",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def bucket_seeds() -> dict[str, list[tuple[int, str | None]]]:
    ledger = {}
    for bucket, relative in BUCKETS.items():
        rows = read_jsonl(ROOT / relative)
        ledger[bucket] = [(row["seed"], row.get("drift")) for row in rows]
    return ledger


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def drift_damage() -> dict[str, dict]:
    """How many points each drift costs the baseline on held-out dev seeds.

    Block 3's exit criterion is "D1-D4 each drop it by at least 20 points".  This
    measures that instead of asserting it.
    """
    from eval.run_agent import run_stale
    from heal.replay import replay
    from world.snapshot import generate_snapshot

    dev_rows = read_jsonl(ROOT / "data" / "dev" / "expert_traces.jsonl")
    seeds = [row["seed"] for row in dev_rows]
    if not seeds:
        return {}

    results = {}
    for drift in (None, "D1", "D2", "D3", "D4"):
        wins = 0
        for seed in seeds:
            trace = run_stale(seed, drift)
            if replay(generate_snapshot(seed), trace["tool_calls"], drift)["correct"]:
                wins += 1
        results[drift or "clean"] = {
            "n": len(seeds),
            "task_success": wins / len(seeds),
        }

    clean = results["clean"]["task_success"]
    for drift, detail in results.items():
        detail["drop_vs_clean"] = clean - detail["task_success"]
        detail["meets_20_point_bar"] = (
            True if drift == "clean" else detail["drop_vs_clean"] >= 0.20
        )

    return results


def stage_drift() -> None:
    """Run the baseline, candidate and poisoned policies on both gate buckets."""
    from eval.run_agent import write_records

    RUNS.mkdir(parents=True, exist_ok=True)
    ledger = bucket_seeds()

    for bucket, entries in ledger.items():
        if not entries:
            print(f"  [skip] {bucket}: no tasks")
            continue
        for policy in ("stale", "expert", "poisoned"):
            path = RUNS / f"{policy}_{bucket}.jsonl"
            records = write_records(entries, policy, bucket, str(path))
            success = sum(1 for row in records if row["success"]) / len(records)
            print(
                f"  {policy:9s} on {bucket:16s} n={len(records):3d} "
                f"task_success={success:.3f}"
            )

    print("\n  drift damage on held-out dev seeds (baseline policy)")
    damage = drift_damage()
    for drift, detail in damage.items():
        bar = "" if drift == "clean" else (
            "  meets 20pt bar" if detail["meets_20_point_bar"]
            else "  BELOW the 20pt bar"
        )
        print(
            f"    {drift:5s} n={detail['n']:3d} "
            f"success={detail['task_success']:.3f} "
            f"drop={detail['drop_vs_clean']:+.3f}{bar}"
        )
    (ARTIFACTS / "drift_damage.json").write_text(
        json.dumps(damage, indent=2), encoding="utf-8"
    )

    print("\n  canaries on dev seeds (no training)")
    table = canaries()
    for name, detail in table.items():
        print(
            f"    {name:18s} success={detail['task_success']:.3f} "
            f"balanced={detail['balanced_success']:.3f} "
            f"(none {detail['success_on_none']:.3f} / "
            f"vuln {detail['success_on_vulnerable']:.3f})"
        )
    (ARTIFACTS / "canaries.json").write_text(
        json.dumps(table, indent=2), encoding="utf-8"
    )


def stage_detect() -> None:
    """Name the failures the baseline actually has, and report residual share."""
    from classifier.rules import classify
    from eval.run_agent import failure_traces

    traces = failure_traces("drift_targeted")
    if not traces:
        print("  [skip] no failures — is drift_targeted drifted?")
        return

    total = len(traces)
    residual = 0
    labels: dict[str, int] = {}
    drifts: dict[str, int] = {}

    for trace in traces:
        result = classify(trace)
        label = result["primary"]
        labels[label] = labels.get(label, 0) + 1
        inferred = result["drift_inferred"] or "none"
        drifts[inferred] = drifts.get(inferred, 0) + 1
        if label == "residual":
            residual += 1

    print(f"  failures: {total}")
    for label, count in sorted(labels.items(), key=lambda item: -item[1]):
        print(f"    {label:20s} {count:4d}  ({count / total:.1%})")
    print(f"  drift inferred from the trace: {drifts}")
    print(f"  residual share: {residual / total:.3f}")
    print("  residual is counted and reported; it is never sent to an LLM.")


def stage_classify() -> None:
    # Rebuild the battery first: it is generated from the live world, so a stale
    # fixture would silently score the classifier against an old environment.
    subprocess.run(
        [sys.executable, "-m", "classifier.build_failure_battery"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "classifier.evaluate_classifier"],
        cwd=ROOT,
        check=True,
    )


def stage_generate() -> None:
    subprocess.run(
        [sys.executable, "-m", "heal.examples", "--drift", "D1"],
        cwd=ROOT,
        check=True,
    )


def stage_retrain() -> None:
    v0 = ROOT / "adapters" / "v0" / "final"
    if not v0.exists():
        print("  [skip] no v0 adapter at adapters/v0/final.")
        print("         Run training/train_v0.py on a GPU first, or use")
        print("         --stage gate to validate the gate on policy records.")
        return
    for script in ("training/train_v0.py", "training/train_heal.py",
                   "training/train_poisoned.py"):
        print(f"  running {script}")
        subprocess.run([sys.executable, script], cwd=ROOT, check=True)


def _records(path: Path, bucket: str) -> list[dict]:
    rows = read_jsonl(path)
    for row in rows:
        row["bucket"] = bucket
    return rows


def stage_gate() -> None:
    from gates.deployment_gate import evaluate_deployment_gate
    from gates.registry import log_decision

    baseline = []
    for bucket in BUCKETS:
        baseline += _records(RUNS / f"stale_{bucket}.jsonl", bucket)

    if not baseline:
        print("  [skip] no baseline records — run --stage drift first")
        return

    for candidate_name in ("expert", "poisoned"):
        candidate = []
        for bucket in BUCKETS:
            candidate += _records(RUNS / f"{candidate_name}_{bucket}.jsonl", bucket)

        result = evaluate_deployment_gate(baseline, candidate)
        log_decision(
            result,
            baseline="stale (v0 policy)",
            candidate=f"{candidate_name} (policy)",
            meta={"note": "policy records, not model rollouts"},
        )

        print(f"\n  candidate: {candidate_name}")
        print(f"  decision:  {result['decision']}")
        print(f"  reason:    {result['reason']}")
        for bucket, detail in result["buckets"].items():
            gain_value = detail["checks"]["task_success_gain"]["value"]
            gain = f"{gain_value:+.3f}" if isinstance(gain_value, (int, float)) else "n/a"
            samples = detail["checks"]["minimum_bucket_samples"]["value"]
            print(
                f"    {bucket:16s} n={samples} "
                f"gain={gain} status={detail['status']}"
            )


def _summary(rows: list[dict]) -> dict:
    import statistics

    success = [row for row in rows if row["success"]]
    steps = [row["num_steps"] for row in success]
    total_calls = sum(row["total_tool_calls"] for row in rows)
    redundant = sum(row["redundant_calls"] for row in rows)

    # Balanced success: mean of success on the `none` tasks and success on the
    # tasks that have a non-empty answer.  Without it, a policy that always says
    # "nothing is wrong" scores the none-rate — roughly 50% — and looks as good
    # as the v0 floor while doing no work at all.
    none_tasks = [row for row in rows if row.get("is_none_task")]
    vuln_tasks = [row for row in rows if row.get("is_none_task") is False]
    none_rate = (
        sum(1 for row in none_tasks if row["success"]) / len(none_tasks)
        if none_tasks else None
    )
    vuln_rate = (
        sum(1 for row in vuln_tasks if row["success"]) / len(vuln_tasks)
        if vuln_tasks else None
    )
    balanced = (
        (none_rate + vuln_rate) / 2
        if none_rate is not None and vuln_rate is not None else None
    )

    return {
        "n": len(rows),
        "task_success": len(success) / len(rows) if rows else 0.0,
        "success_on_none": none_rate,
        "success_on_vulnerable": vuln_rate,
        "balanced_success": balanced,
        "median_steps": statistics.median(steps) if steps else None,
        "redundant_rate": redundant / total_calls if total_calls else None,
    }


def canaries() -> dict[str, dict]:
    """The four degenerate policies the plan says must sit near the floor.

    Measured, not asserted.  `always-none` is the important one: with a ~50%
    none-rate it scores about 50% on task success, which is exactly the v0 bar.
    Balanced success exposes it.
    """
    from heal.replay import replay
    from world.snapshot import generate_snapshot

    dev_rows = read_jsonl(ROOT / "data" / "dev" / "expert_traces.jsonl")
    seeds = [row["seed"] for row in dev_rows]
    if not seeds:
        return {}

    all_ids = []
    for seed in seeds:
        snapshot = generate_snapshot(seed)
        all_ids.append(
            [bucket["id"] for bucket in snapshot["buckets"]]
            + [role["id"] for role in snapshot["roles"]]
            + [group["id"] for group in snapshot["security_groups"]]
        )

    policies = {
        "always-none": lambda index: [{"name": "finish", "arguments": {"answer": []}}],
        "always-all": lambda index: [
            {"name": "finish", "arguments": {"answer": all_ids[index]}}
        ],
        "finish-immediately": lambda index: [
            {"name": "finish", "arguments": {"answer": []}}
        ],
        "repeat-one-call": lambda index: [
            {"name": "list_buckets", "arguments": {}},
            {"name": "list_buckets", "arguments": {}},
            {"name": "list_buckets", "arguments": {}},
            {"name": "finish", "arguments": {"answer": []}},
        ],
    }

    results = {}
    for name, policy in policies.items():
        none_wins = 0
        none_total = 0
        vuln_wins = 0
        vuln_total = 0
        wins = 0
        for index, seed in enumerate(seeds):
            result = replay(generate_snapshot(seed), policy(index), None)
            truth = result["truth"]
            if truth:
                vuln_total += 1
                vuln_wins += int(result["correct"])
            else:
                none_total += 1
                none_wins += int(result["correct"])
            wins += int(result["correct"])

        none_rate = none_wins / none_total if none_total else None
        vuln_rate = vuln_wins / vuln_total if vuln_total else None
        results[name] = {
            "n": len(seeds),
            "task_success": wins / len(seeds),
            "success_on_none": none_rate,
            "success_on_vulnerable": vuln_rate,
            "balanced_success": (
                (none_rate + vuln_rate) / 2
                if none_rate is not None and vuln_rate is not None else None
            ),
        }

    return results


def stage_report() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    conditions = []
    for bucket in BUCKETS:
        for policy in ("stale", "expert", "poisoned"):
            rows = read_jsonl(RUNS / f"{policy}_{bucket}.jsonl")
            if rows:
                conditions.append((bucket, policy, _summary(rows)))

    decisions = read_jsonl(DECISION_LOG)
    manifest_path = ROOT / "data" / "heal" / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else None
    )
    classifier = ROOT / "classifier" / "fixtures" / "classifier_metrics.json"
    classifier_metrics = (
        json.loads(classifier.read_text(encoding="utf-8"))
        if classifier.is_file() else None
    )
    sealed_hash = (ROOT / "data" / "splits" / "sealed_hash.txt").read_text().strip()

    lines = [
        "# HealGuard results",
        "",
        "Generated by `python run_all.py --stage report`. Do not hand-edit.",
        "",
        "> **Provenance.** Every number below is produced by a deterministic",
        "> policy run against the real environment, not by the fine-tuned model.",
        "> The gate, the oracle, the classifier and the replay assertions are the",
        "> same code paths the model records will use; only the policy changes.",
        "> Model records replace these files by path, and the table regenerates.",
        "",
        "## Main table",
        "",
        "| Bucket | Condition | N | Task success | Balanced success | "
        "Success on `none` | Success on vulnerable | Median steps | Redundant rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for bucket, policy, summary in conditions:
        steps = (
            f"{summary['median_steps']:.1f}"
            if summary["median_steps"] is not None else "n/a"
        )
        redundant = (
            f"{summary['redundant_rate']:.3f}"
            if summary["redundant_rate"] is not None else "n/a"
        )
        balanced = (
            f"{summary['balanced_success']:.3f}"
            if summary.get("balanced_success") is not None else "n/a"
        )
        none_rate = (
            f"{summary['success_on_none']:.3f}"
            if summary.get("success_on_none") is not None else "n/a"
        )
        vulnerable_rate = (
            f"{summary['success_on_vulnerable']:.3f}"
            if summary.get("success_on_vulnerable") is not None else "n/a"
        )
        lines.append(
            f"| {bucket} | {policy} | {summary['n']} | "
            f"{summary['task_success']:.3f} | {balanced} | {none_rate} | "
            f"{vulnerable_rate} | {steps} | {redundant} |"
        )

    canary_path = ARTIFACTS / "canaries.json"
    lines += ["", "## Canaries (dev seeds, no training)", ""]
    if canary_path.is_file():
        data = json.loads(canary_path.read_text(encoding="utf-8"))
        lines += [
            "| Policy | N | Task success | Balanced success | "
            "Success on `none` | Success on vulnerable |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, detail in data.items():
            lines.append(
                f"| {name} | {detail['n']} | {detail['task_success']:.3f} | "
                f"{detail['balanced_success']:.3f} | "
                f"{detail['success_on_none']:.3f} | "
                f"{detail['success_on_vulnerable']:.3f} |"
            )
        lines += [
            "",
            "`always-none` scores the none-rate on **task success**, which is why "
            "balanced success is published beside it. The plan's canary bar is a "
            "floor to sit at, not a coincidence to live with.",
        ]
    else:
        lines.append("_Run `--stage drift`._")

    lines += ["", "## Drift damage (baseline policy, held-out dev seeds)", ""]
    damage_path = ARTIFACTS / "drift_damage.json"
    if damage_path.is_file():
        damage = json.loads(damage_path.read_text(encoding="utf-8"))
        lines += [
            "| Condition | N | Task success | Drop vs clean | Meets 20-point bar |",
            "|---|---:|---:|---:|---|",
        ]
        for drift, detail in damage.items():
            bar = (
                "n/a" if drift == "clean"
                else ("yes" if detail["meets_20_point_bar"] else "**no**")
            )
            lines.append(
                f"| {drift} | {detail['n']} | {detail['task_success']:.3f} | "
                f"{detail['drop_vs_clean']:+.3f} | {bar} |"
            )
    else:
        lines.append("_Run `--stage drift`._")

    lines += ["", "## Gate decisions", ""]
    if decisions:
        lines += [
            "| Candidate | Decision | Reason |",
            "|---|---|---|",
        ]
        for record in decisions:
            lines.append(
                f"| {record['candidate']} | {record['decision']} | "
                f"{record['reason']} |"
            )
    else:
        lines.append("_No decisions logged yet._")

    lines += ["", "## Classifier", ""]
    if classifier_metrics:
        battery = classifier_metrics["battery"]
        field = classifier_metrics["field"]
        lines += [
            f"- Constructed battery: n={battery['n']}, "
            f"accuracy={battery['accuracy']:.3f}, "
            f"macro-F1={battery['macro_f1']:.3f}",
        ]
        if field.get("available"):
            lines.append(
                f"- Field rollouts: n={field['n']}, "
                f"accepted={field['accepted_share']:.3f}, "
                f"residual={field['residual_share']:.3f}"
            )
    else:
        lines.append("_Run `--stage classify`._")

    lines += ["", "## Corrective examples", ""]
    if manifest:
        lines += [
            f"- Drift {manifest['drift']}, source `{manifest['source']}`",
            f"- Minimal-edit pairs: {manifest['pairs']}",
            f"- Every chosen trace replays perfectly: "
            f"**{manifest['all_chosen_replay_to_1']}**",
            f"- Clean replay mix: {manifest['clean_mix_share']:.1%}",
            f"- Claimed labels: {', '.join(manifest['claimed_labels']) or 'none'}",
        ]
    else:
        lines.append("_Run `--stage generate`._")

    lines += [
        "",
        "## Reproducibility",
        "",
        f"- Sealed split sha256: `{sealed_hash}`",
        "- The sealed split was not opened to produce any number above.",
        "- Thresholds: `configs/thresholds.yaml` (frozen 11 Nov).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python run_all.py --stage drift     # policy records, no GPU",
        "python run_all.py --stage detect",
        "python run_all.py --stage classify",
        "python run_all.py --stage generate",
        "python run_all.py --stage gate",
        "python run_all.py --stage report",
        "```",
    ]

    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {ARTIFACTS / 'results.md'}")


STAGES = {
    "drift": stage_drift,
    "detect": stage_detect,
    "classify": stage_classify,
    "generate": stage_generate,
    "retrain": stage_retrain,
    "gate": stage_gate,
    "report": stage_report,
}

ORDER = ["drift", "detect", "classify", "generate", "gate", "report"]


def main() -> None:
    parser = argparse.ArgumentParser(description="HealGuard orchestrator")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", *STAGES],
        help="Pipeline stage to run.",
    )
    args = parser.parse_args()

    stages = ORDER if args.stage == "all" else [args.stage]

    for name in stages:
        banner(f"{name}")
        STAGES[name]()
        if args.stage == "all" and name == "drift":
            # `detect` needs the drift records that were just written.
            pass

    banner("done")
    if args.stage in {"all", "report"}:
        print(f"  main table: {ARTIFACTS / 'results.md'}")


if __name__ == "__main__":
    main()
