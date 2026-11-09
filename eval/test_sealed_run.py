"""Sealed-set runner tests. CPU only, no model, no GPU.

The sealed set itself is never opened here beyond reading its task list: the
tests use the first few tasks and a scripted backend.
"""

import json
import unittest
from pathlib import Path

from eval.rollout import ScriptedBackend
from eval.sealed_run import CONDITIONS, evaluate, markdown, summarise, wilson

ROOT = Path(__file__).resolve().parents[1]


def call_text(name, arguments):
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


class TestWilson(unittest.TestCase):

    def test_interval_brackets_the_estimate(self):
        low, high = wilson(25, 50)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_small_n_is_wide(self):
        low_small, high_small = wilson(3, 6)
        low_big, high_big = wilson(25, 50)
        self.assertGreater(high_small - low_small, high_big - low_big)

    def test_perfect_score_stays_below_one(self):
        low, high = wilson(5, 5)
        self.assertEqual(high, 1.0)
        self.assertLess(low, 1.0)

    def test_empty_is_zero(self):
        self.assertEqual(wilson(0, 0), (0.0, 0.0))


class TestSummarise(unittest.TestCase):

    def test_balanced_success_is_the_mean_of_the_two_halves(self):
        records = [
            {"success": True, "is_none_task": True, "num_steps": 4,
             "total_tool_calls": 5, "redundant_calls": 0},
            {"success": True, "is_none_task": False, "num_steps": 6,
             "total_tool_calls": 7, "redundant_calls": 0},
            {"success": False, "is_none_task": False, "num_steps": 9,
             "total_tool_calls": 10, "redundant_calls": 1},
        ]
        row = summarise("x", records)

        self.assertEqual(row["task_success"], 2 / 3)
        self.assertEqual(row["success_on_none"], 1.0)
        self.assertEqual(row["success_on_vulnerable"], 0.5)
        self.assertEqual(row["balanced_success"], 0.75)
        # steps are taken over successes only
        self.assertEqual(row["median_steps"], 5.0)
        self.assertAlmostEqual(row["redundant_rate"], 1 / 22)


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        from data.splits.sealed import load_sealed_tasks

        self.tasks = load_sealed_tasks()[:4]

    def test_expert_is_perfect_and_needs_no_backend(self):
        called = []

        def backend_for(adapter):
            called.append(adapter)
            return ScriptedBackend([call_text("finish", {"answer": []})])

        results = evaluate(backend_for, self.tasks, ["expert"])

        self.assertEqual(results["expert"]["task_success"], 1.0)
        self.assertEqual(called, [], "the expert must not load a model")
        self.assertIsNotNone(results["expert"]["median_steps"])

    def test_a_model_condition_is_scored_on_the_same_tasks(self):
        def backend_for(adapter):
            return ScriptedBackend([call_text("finish", {"answer": []})])

        results = evaluate(backend_for, self.tasks, ["expert", "v0_d1_stale"])

        self.assertEqual(results["v0_d1_stale"]["n"], len(self.tasks))
        # always answering [] scores every none task and no vulnerable one
        self.assertEqual(results["v0_d1_stale"]["success_on_none"], 1.0)
        self.assertEqual(results["v0_d1_stale"]["success_on_vulnerable"], 0.0)

    def test_each_adapter_is_loaded_once(self):
        loads = []

        def backend_for(adapter):
            loads.append(adapter)
            return ScriptedBackend([call_text("finish", {"answer": []})])

        evaluate(
            backend_for,
            self.tasks,
            ["v0_clean", "v0_d1_stale", "v0_d1_prompt_only"],
        )

        self.assertEqual(len(loads), 1, loads)

    def test_limit_takes_a_prefix(self):
        def backend_for(adapter):
            return ScriptedBackend([call_text("finish", {"answer": []})])

        results = evaluate(backend_for, self.tasks, ["v0_d1_stale"], limit=2)
        self.assertEqual(results["v0_d1_stale"]["n"], 2)


class TestMarkdown(unittest.TestCase):

    def test_table_lists_every_condition_in_order(self):
        results = {
            name: {
                "condition": name,
                "n": 50,
                "task_success": 0.5,
                "ci95": [0.36, 0.64],
                "success_on_none": 1.0,
                "success_on_vulnerable": 0.0,
                "balanced_success": 0.5,
                "median_steps": 12.0,
                "redundant_rate": 0.0,
                "mean_calls": 12.0,
            }
            for name in ("expert", "v0_d1_stale")
        }
        table = markdown(results, "abc123", none_tasks=25, total=50)

        self.assertIn("| expert |", table)
        self.assertIn("| v0_d1_stale |", table)
        self.assertIn("[0.36, 0.64]", table)
        self.assertIn("25 `none`", table)
        self.assertIn("one seed", table)
        self.assertLess(table.index("expert"), table.index("v0_d1_stale"))


class TestConditions(unittest.TestCase):

    def test_every_condition_has_a_complete_spec(self):
        for name, spec in CONDITIONS.items():
            with self.subTest(name=name):
                if spec.get("kind") == "expert":
                    continue
                self.assertIn("adapter", spec)
                self.assertIn("drift", spec)
                self.assertIn("stale", spec)

    def test_the_healed_candidate_and_the_baseline_use_the_same_condition(self):
        baseline = CONDITIONS["v0_d1_stale"]
        candidate = CONDITIONS["v1_good_d1_stale"]
        self.assertEqual(baseline["drift"], candidate["drift"])
        self.assertEqual(baseline["stale"], candidate["stale"])


if __name__ == "__main__":
    unittest.main()
