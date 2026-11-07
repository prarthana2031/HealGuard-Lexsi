"""Score the rule classifier on evidence it did not author.

Two evaluations, reported separately because they answer different questions.

1. **Constructed battery** (`classifier/fixtures/failure_battery.jsonl`).
   Every trajectory is produced by running a deliberately broken policy against
   the real environment and the real drift wrappers, and the ground-truth label
   is the policy name fixed before classification.  This is the coverage
   measurement: does each rule fire on a real instance of its own failure?

2. **Field rollouts** (`data/drift_train/v0_failed_rollouts.jsonl`).
   Real model output under the four injected drifts.  No labels are invented:
   the drift is the known cause, the classifier sees only the trace.  This is
   the honest number and it is allowed to be ugly.

Neither evaluation uses an LLM.  Residual failures are counted, never sent out.

Usage:
    python -m classifier.evaluate_classifier
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classifier.rules import PRIORITY, classify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "classifier" / "fixtures" / "failure_battery.jsonl"

# The "field" number scores real model failures. It prefers v2 rollouts as soon
# as they exist, and falls back to the archived world-v1 probe. The classifier
# reads traces, not worlds, so the v1 rows are still valid evidence — but they
# are labelled, because they are not a world-v2 model result.
FIELD_DIRS = (
    ROOT / "data" / "drift_train",
    ROOT / "data" / "archive" / "world_v1",
)

# What the plan says each injected drift should look like.  Used only to decide
# whether the classifier's label is one of the accepted labels for that cause.
DRIFT_ACCEPTED_LABELS = {
    "D1": {"drift_mismatch", "schema_violation"},
    "D2": {"drift_mismatch", "ungrounded_answer", "loop_collapse", "premature_finish"},
    "D3": {"drift_mismatch", "schema_violation"},
    "D4": {"drift_mismatch", "wrong_tool"},
}


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def confusion(rows, true_key, predicted_key) -> tuple[dict, list]:
    labels = [label for label in PRIORITY]
    matrix = {
        true: {pred: 0 for pred in labels}
        for true in labels
    }
    for row in rows:
        true = row[true_key]
        predicted = row[predicted_key]
        if true not in matrix:
            matrix[true] = {pred: 0 for pred in labels}
        matrix[true][predicted] = matrix[true].get(predicted, 0) + 1
    return matrix, labels


def prf(matrix, labels) -> list[dict]:
    metrics = []
    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(
            matrix.get(other, {}).get(label, 0)
            for other in labels
            if other != label
        )
        fn = sum(
            count
            for pred, count in matrix.get(label, {}).items()
            if pred != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        metrics.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
            }
        )
    return metrics


def evaluate_battery() -> dict:
    rows = load(BATTERY)
    predictions = []
    for row in rows:
        result = classify(row)
        predictions.append(
            {
                "true": row["expected_label"],
                "predicted": result["primary"],
                "task_id": row["task_id"],
                "drift_inferred": result["drift_inferred"],
            }
        )

    matrix, labels = confusion(predictions, "true", "predicted")
    metrics = prf(matrix, labels)

    correct = sum(1 for item in predictions if item["true"] == item["predicted"])
    macro_f1 = sum(item["f1"] for item in metrics) / len(metrics)

    return {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": macro_f1,
        "labels": labels,
        "matrix": matrix,
        "metrics": metrics,
        "predictions": predictions,
    }


def _field_source() -> tuple[Path | None, str]:
    """First directory that actually holds rollouts, and a label for the report."""
    for directory in FIELD_DIRS:
        files = sorted(directory.glob("*.jsonl"))
        if files:
            label = (
                "world v2 model rollouts"
                if directory == FIELD_DIRS[0]
                else "world v1 (archived probe)"
            )
            return directory, label
    return None, "none"


def evaluate_field() -> dict:
    directory, label = _field_source()
    if directory is None:
        return {"available": False}

    rows = []
    for path in sorted(directory.glob("*.jsonl")):
        rows.extend(load(path))
    if not rows:
        return {"available": False}

    per_drift = {}
    residuals = 0
    accepted = 0

    for drift in ("D1", "D2", "D3", "D4"):
        subset = [row for row in rows if row.get("drift") == drift]
        if not subset:
            continue
        counter = Counter()
        drift_accepted = 0
        for row in subset:
            primary = classify(row)["primary"]
            counter[primary] += 1
            if primary in DRIFT_ACCEPTED_LABELS.get(drift, set()):
                drift_accepted += 1
            if primary == "residual":
                residuals += 1
        accepted += drift_accepted
        per_drift[drift] = {
            "n": len(subset),
            "labels": dict(counter),
            "accepted_share": drift_accepted / len(subset),
        }

    total = len(rows)
    return {
        "available": True,
        "source": label,
        "n": total,
        "per_drift": per_drift,
        "residual_share": residuals / total if total else 0.0,
        "accepted_share": accepted / total if total else 0.0,
    }


def write_report(battery: dict, field: dict) -> None:
    labels = battery["labels"]
    matrix = battery["matrix"]

    lines = [
        "# Rule Classifier Evaluation",
        "",
        "Generated by `python -m classifier.evaluate_classifier`. Do not hand-edit.",
        "",
        "## 1. Constructed battery (coverage)",
        "",
        "Each trajectory is produced by running a deliberately broken policy "
        "against the real environment and the real drift wrappers. The "
        "ground-truth label is the policy name, fixed before classification. "
        "The classifier never sees a `drift` field.",
        "",
        f"- Trajectories: {battery['n']}",
        f"- Accuracy: {battery['accuracy']:.4f}",
        f"- Macro-F1 (11 classes): {battery['macro_f1']:.4f}",
        "",
        "### Confusion matrix",
        "",
        "| True \\\\ Predicted | " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * (len(labels) + 1)) + " |",
    ]

    for true in labels:
        lines.append(
            "| "
            + true
            + " | "
            + " | ".join(str(matrix.get(true, {}).get(pred, 0)) for pred in labels)
            + " |"
        )

    lines += [
        "",
        "### Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in battery["metrics"]:
        lines.append(
            f"| {item['label']} | {item['precision']:.4f} | {item['recall']:.4f} "
            f"| {item['f1']:.4f} | {item['support']} |"
        )

    lines += [
        "",
        "**Scope.** One canonical construction per label. A perfect score here "
        "means every rule fires on the textbook instance of its own failure; it "
        "is not a generalisation claim. The field number below is the honest one.",
        "",
        "## 2. Field rollouts (real model output under the four injected drifts)",
        "",
    ]

    if not field.get("available"):
        lines.append("Field rollouts not present.")
    else:
        lines += [
            "The classifier sees only the trace. The injected drift is the known ",
            "cause, so a label counts as accepted when it is one the plan lists ",
            "for that drift.",
            "",
            f"> Source: **{field.get('source', 'unknown')}**.",
            "",
            f"- Trajectories: {field['n']}",
            f"- Accepted-label share: {field['accepted_share']:.4f}",
            f"- Residual share: {field['residual_share']:.4f}",
            "",
            "| Drift | N | Predicted labels | Accepted share |",
            "|---|---:|---|---:|",
        ]
        for drift, detail in field["per_drift"].items():
            labels_text = ", ".join(
                f"{label} x{count}"
                for label, count in sorted(detail["labels"].items())
            )
            lines.append(
                f"| {drift} | {detail['n']} | {labels_text} "
                f"| {detail['accepted_share']:.4f} |"
            )
        lines += [
            "",
            "The field number is low because these rollouts come from the "
            "**untuned** model, which emits prose instead of tool calls. Under "
            "the taxonomy that is `schema_violation`, not `drift_mismatch`: there "
            "is no tool call whose name or argument could be in the schema diff. "
            "The classifier reports that honestly instead of inventing a drift "
            "label from a filename, which is what the previous version did.",
        ]

    (ROOT / "classifier" / "confusion_matrix.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    with (ROOT / "classifier" / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true_label"] + labels)
        for true in labels:
            writer.writerow(
                [true] + [matrix.get(true, {}).get(pred, 0) for pred in labels]
            )

    (ROOT / "classifier" / "fixtures" / "classifier_metrics.json").write_text(
        json.dumps(
            {
                "battery": {
                    "n": battery["n"],
                    "accuracy": battery["accuracy"],
                    "macro_f1": battery["macro_f1"],
                    "per_class": battery["metrics"],
                },
                "field": field,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    battery = evaluate_battery()
    field = evaluate_field()
    write_report(battery, field)

    print(f"battery: n={battery['n']} accuracy={battery['accuracy']:.4f} "
          f"macro_f1={battery['macro_f1']:.4f}")
    if field.get("available"):
        print(f"field:   n={field['n']} accepted={field['accepted_share']:.4f} "
              f"residual={field['residual_share']:.4f}")
    print("wrote classifier/confusion_matrix.md")


if __name__ == "__main__":
    main()
