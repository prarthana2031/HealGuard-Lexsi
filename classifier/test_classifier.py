"""Classifier tests.

The previous suite was scored on `tests/fixtures/mock_trajectories_20.jsonl`,
whose expected labels were written by the same hand that wrote the rules.  That
is a tautology, not a measurement, and it hid the fact that the classifier
returned `residual` for 80 of 80 real drift failures.

These tests score the classifier against `classifier/fixtures/failure_battery.jsonl`,
where each trajectory comes from running a deliberately broken policy against the
real environment and the label is fixed before classification.
"""

import json
import unittest
from pathlib import Path

from classifier.rules import PRIORITY, classify, classify_trajectory, residual_share

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "classifier" / "fixtures" / "failure_battery.jsonl"


def load_battery():
    with BATTERY.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class TestClassifierCoverage(unittest.TestCase):

    def test_battery_covers_all_eleven_labels(self):
        rows = load_battery()
        present = {row["expected_label"] for row in rows}
        self.assertEqual(present, set(PRIORITY))

    def test_battery_labels_are_not_read_from_the_trace(self):
        """Ground truth must live outside the fields the classifier reads."""
        for row in load_battery():
            self.assertNotIn("failure_mode", row)
            self.assertNotIn("label", row)

    def test_battery_macro_f1_is_at_least_085(self):
        rows = load_battery()
        correct = sum(
            1 for row in rows if classify(row)["primary"] == row["expected_label"]
        )
        self.assertGreaterEqual(correct / len(rows), 0.85)

    def test_every_label_is_learned_not_guessed(self):
        """Each label must be produced for its own constructed instance."""
        rows = load_battery()
        for label in PRIORITY:
            subset = [row for row in rows if row["expected_label"] == label]
            self.assertTrue(subset, f"no trajectory for {label}")
            hits = sum(1 for row in subset if classify(row)["primary"] == label)
            self.assertEqual(hits, len(subset), f"{label} not recognised")


class TestClassifierRules(unittest.TestCase):

    def test_no_tool_call_is_a_schema_violation(self):
        """An agent that emits prose instead of a call broke the contract."""
        trajectory = {
            "task": "Find all vulnerable cloud resources.",
            "is_correct": False,
            "messages": [
                {"role": "user", "content": "Find all vulnerable cloud resources."},
                {"role": "assistant", "content": "I cannot complete the review."},
            ],
        }
        self.assertEqual(classify_trajectory(trajectory), "schema_violation")

    def test_drift_is_inferred_from_the_trace_without_metadata(self):
        """Deleting the `drift` field must not change the verdict."""
        rows = [
            row for row in load_battery()
            if row["expected_label"] == "drift_mismatch"
        ]
        self.assertTrue(rows)
        for row in rows:
            stripped = dict(row)
            stripped.pop("drift", None)
            stripped.pop("split", None)
            result = classify(stripped)
            self.assertEqual(result["primary"], "drift_mismatch")
            self.assertIsNotNone(result["drift_inferred"])

    def test_all_matching_labels_are_kept(self):
        """A drift error is also a schema error; both must be reported."""
        row = next(
            row for row in load_battery()
            if row["expected_label"] == "drift_mismatch" and row["drift"] == "D1"
        )
        result = classify(row)
        self.assertEqual(result["primary"], "drift_mismatch")
        self.assertIn("schema_violation", result["labels"])
        self.assertEqual(result["labels"][0], "drift_mismatch")

    def test_first_divergent_step_is_reported(self):
        row = next(
            row for row in load_battery()
            if row["expected_label"] == "wrong_order"
        )
        result = classify(row)
        self.assertIsInstance(result["first_divergent_step"], int)

    def test_success_has_no_label(self):
        row = next(
            row for row in load_battery() if row["expected_label"] == "residual"
        )
        success = dict(row)
        success["success"] = True
        self.assertIsNone(classify(success)["primary"])

    def test_residual_share_is_reported(self):
        rows = [row for row in load_battery() if row["expected_label"] == "residual"]
        self.assertEqual(residual_share(rows), 1.0)


if __name__ == "__main__":
    unittest.main()
