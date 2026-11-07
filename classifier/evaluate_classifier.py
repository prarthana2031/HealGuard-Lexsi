import csv
import json
from collections import Counter
from pathlib import Path

from classifier.rules import PRIORITY, classify_trajectory

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EXPECTED_FILE = FIXTURES / "expected_labels.json"
MOCK_FILE = FIXTURES / "mock_trajectories_20.jsonl"

with EXPECTED_FILE.open(encoding="utf-8") as f:
    expected = json.load(f)

with MOCK_FILE.open(encoding="utf-8") as f:
    trajectories = [json.loads(line) for line in f if line.strip()]

actual_ids = {t["task_id"] for t in trajectories}
missing_ids = sorted(set(expected) - actual_ids)

rows = []
for trajectory in trajectories:
    task_id = trajectory["task_id"]
    if task_id not in expected:
        raise ValueError(f"Missing expected label for {task_id}")

    true_label = expected[task_id]["label"]
    predicted_label = classify_trajectory(trajectory)
    rows.append((task_id, true_label, predicted_label))

def display(label):
    return "no_failure" if label is None else label

observed = {label for _, true, pred in rows for label in (true, pred)}
labels = [label for label in PRIORITY if label in observed]
if None in observed:
    labels.append(None)

matrix = {
    true: {
        pred: sum(1 for _, t, p in rows if t == true and p == pred)
        for pred in labels
    }
    for true in labels
}

metrics = []
for label in labels:
    tp = matrix[label][label]
    fp = sum(matrix[other][label] for other in labels if other != label)
    fn = sum(matrix[label][other] for other in labels if other != label)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    support = sum(matrix[label].values())
    metrics.append((label, precision, recall, f1, support))

accuracy = (
    sum(1 for _, true, pred in rows if true == pred) / len(rows)
    if rows else 0.0
)
macro_f1 = (
    sum(item[3] for item in metrics) / len(metrics)
    if metrics else 0.0
)

# Save confusion matrix CSV.
csv_path = ROOT / "classifier" / "confusion_matrix.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["true_label"] + [display(x) for x in labels])
    for true in labels:
        writer.writerow(
            [display(true)] + [matrix[true][pred] for pred in labels]
        )

# Save report.
report = [
    "# Rule Classifier Evaluation",
    "",
    f"- Trajectories evaluated: {len(rows)}",
    f"- Accuracy: {accuracy:.4f}",
    f"- Macro-F1 (represented classes): {macro_f1:.4f}",
    f"- Expected-label entries: {len(expected)}",
    f"- Expected entries not present in mock file: {len(missing_ids)}",
    "",
    "## Confusion matrix",
    "",
    "| True \\\\ Predicted | " + " | ".join(display(x) for x in labels) + " |",
    "| " + " | ".join(["---"] * (len(labels) + 1)) + " |",
]

for true in labels:
    values = [str(matrix[true][pred]) for pred in labels]
    report.append(
        "| " + display(true) + " | " + " | ".join(values) + " |"
    )

report += [
    "",
    "## Per-class metrics",
    "",
    "| Class | Precision | Recall | F1 | Support |",
    "|---|---:|---:|---:|---:|",
]

for label, precision, recall, f1, support in metrics:
    report.append(
        f"| {display(label)} | {precision:.4f} | {recall:.4f} "
        f"| {f1:.4f} | {support} |"
    )

represented = {display(label) for label in labels}
unrepresented = [
    label for label in PRIORITY if display(label) not in represented
]

report += [
    "",
    "## Coverage limitations",
    "",
    "The macro-F1 above includes only classes represented in the "
    "fixture's true or predicted labels. It is not a full 11-class "
    "validation score.",
    "",
    "Taxonomy labels without fixture coverage: "
    + (", ".join(unrepresented) if unrepresented else "None"),
    "",
    "Expected-label IDs absent from the mock trajectories: "
    + (", ".join(missing_ids) if missing_ids else "None"),
    "",
    "This is a fixture evaluation, not the final injected-drift "
    "macro-F1 measurement.",
]

md_path = ROOT / "classifier" / "confusion_matrix.md"
md_path.write_text("\n".join(report) + "\n", encoding="utf-8")

print(f"Trajectories evaluated: {len(rows)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1 (represented classes): {macro_f1:.4f}")
print(f"Missing expected-label entries: {len(missing_ids)}")
print(f"Classes without fixture coverage: {', '.join(unrepresented)}")
print(f"Saved: {md_path}")
print(f"Saved: {csv_path}")
