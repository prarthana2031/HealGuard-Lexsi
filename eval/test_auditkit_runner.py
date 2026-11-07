
import unittest

from eval.auditkit_runner import evaluate_trajectories, load_trajectories


class TestAuditKitRunner(unittest.TestCase):

    def make_trajectory(
        self,
        trajectory_id,
        success,
        valid_calls,
        total_calls,
        redundant_calls,
        steps,
    ):
        return {
            "id": trajectory_id,
            "seed": 1,
            "split": "dev",
            "task": "List buckets.",
            "messages": [
                {"role": "system", "content": "You are a cloud auditor."},
                {"role": "user", "content": "List buckets."},
                {
                    "role": "assistant",
                    "content": (
                        '<tool_call>\n'
                        '{"name": "list_buckets", "arguments": {}}\n'
                        '</tool_call>'
                    ),
                },
                {
                    "role": "tool",
                    "content": '{"buckets": ["bucket-1"]}',
                },
                {
                    "role": "assistant",
                    "content": "The bucket is bucket-1.",
                },
            ],
            "tool_calls": [
                {
                    "name": "list_buckets",
                    "arguments": {},
                    "valid": True,
                }
            ],
            "reward": 1.0 if success else 0.0,
            "success": success,
            "num_steps": steps,
            "template_id": "custom_tool_call_v1",
            "error": None,
            "valid_tool_calls": valid_calls,
            "total_tool_calls": total_calls,
            "redundant_calls": redundant_calls,
            "finished": True,
        }

    def test_dataset_level_aggregation(self):
        trajectories = [
            self.make_trajectory(
                "t1", True, 9, 10, 1, 4
            ),
            self.make_trajectory(
                "t2", True, 1, 1, 0, 2
            ),
        ]

        result = evaluate_trajectories(trajectories)
        metrics = result["metrics"]

        self.assertEqual(
            metrics["task_success"]["value"], 1.0
        )
        self.assertAlmostEqual(
            metrics["valid_call_rate"]["value"], 10 / 11
        )
        self.assertAlmostEqual(
            metrics["redundant_rate"]["value"], 1 / 11
        )
        self.assertEqual(
            metrics["steps_on_success"]["value"], 3.0
        )

    def test_loop_rate_uses_total_trajectories(self):
        looping = self.make_trajectory(
            "loop-1", False, 3, 3, 2, 3
        )

        call = {
            "name": "list_buckets",
            "arguments": {},
            "valid": True,
        }

        call_content = (
            '<tool_call>\n'
            '{"name": "list_buckets", "arguments": {}}\n'
            '</tool_call>'
        )

        looping["messages"] = [
            {"role": "system", "content": "You are a cloud auditor."},
            {"role": "user", "content": "List buckets."},
        ]

        for _ in range(3):
            looping["messages"].append({
                "role": "assistant",
                "content": call_content,
            })
            looping["messages"].append({
                "role": "tool",
                "content": '{"buckets": ["bucket-1"]}',
            })

        looping["tool_calls"] = [call, call, call]

        normal = self.make_trajectory(
            "normal-1", True, 1, 1, 0, 2
        )

        result = evaluate_trajectories([looping, normal])

        self.assertEqual(
            result["metrics"]["loop_rate"]["denominator"], 2
        )
        self.assertAlmostEqual(
            result["metrics"]["loop_rate"]["value"], 0.5
        )


    def test_empty_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_trajectories([])

    def test_zero_tool_calls_return_na(self):
        trajectory = self.make_trajectory(
            "t1", False, 0, 0, 0, 0
        )

        result = evaluate_trajectories([trajectory])
        metrics = result["metrics"]

        self.assertIsNone(
            metrics["valid_call_rate"]["value"]
        )
        self.assertIsNone(
            metrics["redundant_rate"]["value"]
        )

    def test_failed_trajectories_do_not_affect_success_steps(self):
        trajectories = [
            self.make_trajectory(
                "t1", True, 1, 1, 0, 4
            ),
            self.make_trajectory(
                "t2", False, 1, 1, 0, 100
            ),
        ]

        result = evaluate_trajectories(trajectories)

        self.assertEqual(
            result["metrics"]["steps_on_success"]["value"],
            4.0,
        )
    def test_steps_on_success_uses_median(self):
        trajectories = [
            self.make_trajectory("t1", True, 1, 1, 0, 2),
            self.make_trajectory("t2", True, 1, 1, 0, 4),
            self.make_trajectory("t3", True, 1, 1, 0, 100),
        ]

        result = evaluate_trajectories(trajectories)

        self.assertEqual(
            result["metrics"]["steps_on_success"]["value"],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
