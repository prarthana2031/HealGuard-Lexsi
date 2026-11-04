
import unittest

from eval.metrics import (
    METRICS,
    trajectory_to_sample,
)


class TestHealGuardMetrics(unittest.TestCase):

    def setUp(self):
        self.trajectory = {
            "id": "traj-00042",
            "seed": 42,
            "split": "dev",
            "task": "Is bucket public-read allowed?",
            "messages": [
                {"role": "system", "content": "..."},
                {
                    "role": "user",
                    "content": "Is bucket public-read allowed?",
                },
                {
                    "role": "assistant",
                    "content": (
                        '<tool_call>\n'
                        '{"name": "check_bucket_acl", '
                        '"arguments": {"bucket": "my-bucket"}}\n'
                        '</tool_call>'
                    ),
                },
                {"role": "tool", "content": '{"public": false}'},
                {
                    "role": "assistant",
                    "content": "No, the bucket is not public-read.",
                },
            ],
            "tool_calls": [
                {
                    "name": "check_bucket_acl",
                    "arguments": {"bucket": "my-bucket"},
                    "valid": True,
                }
            ],
            "reward": 1.0,
            "success": True,
            "num_steps": 2,
            "template_id": "custom_tool_call_v1",
            "error": None,
            "valid_tool_calls": 1,
            "total_tool_calls": 1,
            "redundant_calls": 0,
            "finished": True,
        }

        self.sample, self.output, self.context = (
            trajectory_to_sample(self.trajectory)
        )

    def test_all_five_scorers_registered(self):
        expected = [
            "task_success",
            "valid_call_rate",
            "redundant_rate",
            "loop_rate",
            "steps_on_success",
        ]

        self.assertEqual(
            [metric.name for metric in METRICS],
            expected,
        )

    def test_task_success(self):
        score = METRICS[0](
            self.sample, self.output, context=self.context
        )
        self.assertEqual(score.value, 1.0)

    def test_valid_call_rate(self):
        score = METRICS[1](
            self.sample, self.output, context=self.context
        )
        self.assertEqual(score.value, 1.0)

    def test_redundant_rate(self):
        score = METRICS[2](
            self.sample, self.output, context=self.context
        )
        self.assertEqual(score.value, 0.0)

    def test_loop_rate(self):
        score = METRICS[3](
            self.sample, self.output, context=self.context
        )
        self.assertEqual(score.value, 0.0)
        self.assertFalse(score.metadata["loop_detected"])

    def test_steps_on_success(self):
        score = METRICS[4](
            self.sample, self.output, context=self.context
        )
        self.assertEqual(score.value, 2.0)

    def test_zero_tool_calls_are_not_applicable(self):
        trajectory = dict(self.trajectory)
        trajectory["tool_calls"] = []
        trajectory["total_tool_calls"] = 0
        trajectory["valid_tool_calls"] = 0
        trajectory["redundant_calls"] = 0

        sample, output, context = trajectory_to_sample(trajectory)

        valid_score = METRICS[1](
            sample, output, context=context
        )
        redundant_score = METRICS[2](
            sample, output, context=context
        )

        self.assertTrue(valid_score.metadata["not_applicable"])
        self.assertTrue(redundant_score.metadata["not_applicable"])


if __name__ == "__main__":
    unittest.main()
