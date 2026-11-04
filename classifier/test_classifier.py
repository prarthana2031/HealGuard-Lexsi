
import json
import unittest
from pathlib import Path

from classifier.rules import classify_trajectory


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

EXPECTED_LABELS = json.loads(
    (FIXTURES / "expected_labels.json").read_text(encoding="utf-8")
)

MOCK_FILE = FIXTURES / "mock_trajectories_20.jsonl"


def load_mock_trajectories():
    with MOCK_FILE.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


class TestClassifier(unittest.TestCase):

    def test_expected_labels_file_has_20_entries(self):
        self.assertEqual(len(EXPECTED_LABELS), 20)

    def test_mock_file_has_16_trajectories(self):
        trajectories = load_mock_trajectories()
        self.assertEqual(len(trajectories), 16)

    def test_all_mock_trajectories_have_expected_labels(self):
        trajectories = load_mock_trajectories()

        for trajectory in trajectories:
            task_id = trajectory["task_id"]

            with self.subTest(task_id=task_id):
                self.assertIn(task_id, EXPECTED_LABELS)

    def test_classifier_returns_valid_label(self):
        trajectories = load_mock_trajectories()

        valid_labels = {
            None,
            "env_error",
            "drift_mismatch",
            "schema_violation",
            "wrong_tool",
            "wrong_order",
            "premature_finish",
            "ungrounded_answer",
            "loop_collapse",
            "redundant_call",
            "unrecovered_error",
            "residual",
        }

        for trajectory in trajectories:
            with self.subTest(task_id=trajectory["task_id"]):
                result = classify_trajectory(trajectory)
                self.assertIn(result, valid_labels)

    def test_predictions_match_expected_labels(self):
        trajectories = load_mock_trajectories()

        for trajectory in trajectories:
            task_id = trajectory["task_id"]

            expected = EXPECTED_LABELS[task_id]["label"]
            actual = classify_trajectory(trajectory)

            with self.subTest(task_id=task_id):
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
