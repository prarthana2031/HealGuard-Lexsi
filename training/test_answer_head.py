"""Tests for the answer-head stage. CPU only, no model.

The property that matters is that the (prompt, completion) pair is byte-identical
to rendering the whole conversation: otherwise this stage trains on a prompt the
model never sees at inference, which is the class of bug that produced a 0.35
model while every test was green.
"""

import json
import unittest
from pathlib import Path

from training.answer_head import answer_index, answer_set, build_examples
from training.template import TOOL_TEMPLATE_NO_MARKERS

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train_clean" / "expert_traces.jsonl"


def load_rows(limit: int = 20) -> list[dict]:
    with TRAIN.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()][:limit]


def make_renderer(bos: str = "<BOS>"):
    """Render the real template with plain jinja2, markers stripped."""
    from jinja2 import Environment

    template = Environment().from_string(TOOL_TEMPLATE_NO_MARKERS)

    def render(messages):
        return template.render(
            messages=messages, add_generation_prompt=False, bos_token=bos
        )

    return render


def call_text(name, arguments):
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


class TestAnswerIndex(unittest.TestCase):

    def test_finds_the_finish_turn(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": call_text("list_buckets", {})},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": call_text("finish", {"answer": ["a"]})},
            {"role": "tool", "content": "{}"},
        ]
        self.assertEqual(answer_index(messages), 4)
        self.assertEqual(answer_set(messages[4]), ["a"])

    def test_missing_finish_is_none(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "I cannot do that."},
        ]
        self.assertIsNone(answer_index(messages))


class TestBuildExamples(unittest.TestCase):

    def setUp(self):
        self.rows = load_rows()
        self.render = make_renderer()
        self.examples, self.stats = build_examples(
            self.rows, self.render, bos="<BOS>"
        )

    def test_every_trace_produces_an_example(self):
        self.assertEqual(self.stats["skipped"], 0)
        self.assertEqual(len(self.examples), len(self.rows))

    def test_prompt_ends_on_a_tool_result(self):
        """The model must be asked to answer *after* the evidence."""
        for example in self.examples:
            self.assertIn("Tool result:", example["prompt"][-4000:])
            self.assertNotIn("finish", example["prompt"].split("<|END_OF_TURN_TOKEN|>")[-2])

    def test_completion_is_the_finish_call(self):
        for example in self.examples:
            self.assertIn("finish", example["completion"])
            self.assertNotIn("Tool result:", example["completion"])

    def test_prompt_ends_where_generation_starts(self):
        """Inference leaves the prompt at START_RESPONSE; training must too."""
        for example in self.examples:
            self.assertTrue(
                example["prompt"].endswith("<|START_RESPONSE|>"),
                example["prompt"][-120:],
            )

    def test_completion_has_no_turn_header(self):
        """Supervising the header would teach the model to emit a second one."""
        for example in self.examples:
            self.assertNotIn("<|START_OF_TURN_TOKEN|>", example["completion"])
            self.assertNotIn("<|CHATBOT_TOKEN|>", example["completion"])
            self.assertNotIn("<|START_RESPONSE|>", example["completion"])

    def test_completion_ends_with_the_end_of_turn_tokens(self):
        for example in self.examples:
            self.assertTrue(example["completion"].endswith("<|END_OF_TURN_TOKEN|>"))

    def test_prompt_plus_completion_equals_the_whole_conversation(self):
        """The parity check the builder enforces, asserted independently here.

        Examples are shuffled, so match on seed rather than position.
        """
        by_seed = {row["seed"]: row for row in self.rows}
        checked = 0
        for example in self.examples:
            row = by_seed[example["seed"]]
            messages = row["messages"]
            index = answer_index(messages)
            full = self.render(messages[: index + 1])
            self.assertEqual(example["prompt"] + example["completion"], full)
            checked += 1
        self.assertEqual(checked, len(self.examples))

    def test_both_answer_classes_are_present(self):
        self.assertGreater(self.stats["none"], 0)
        self.assertGreater(self.stats["vulnerable"], 0)

    def test_answers_are_parsed(self):
        for example in self.examples:
            kind = "none" if not example["answer"] else "vulnerable"
            self.assertEqual(example["answer_kind"], kind)


class TestBosHandling(unittest.TestCase):

    def test_prefix_is_derived_from_the_template_when_not_given(self):
        """The BOS prefix must come from the template, not the tokenizer."""
        render = make_renderer(bos="<BOS>")
        examples, stats = build_examples(load_rows(limit=5), render)
        self.assertEqual(stats["skipped"], 0)
        for example in examples:
            self.assertFalse(example["completion"].startswith("<BOS>"))
            self.assertTrue(example["prompt"].startswith("<BOS>"))


class TestRepeatVulnerable(unittest.TestCase):

    def test_doubling_affects_only_non_empty_answers(self):
        rows = load_rows()
        render = make_renderer()
        base, base_stats = build_examples(rows, render, bos="<BOS>")
        twice, twice_stats = build_examples(
            rows, render, bos="<BOS>", repeat_vulnerable=2
        )

        self.assertEqual(twice_stats["none"], base_stats["none"])
        self.assertEqual(twice_stats["vulnerable"], base_stats["vulnerable"] * 2)
        self.assertGreater(len(twice), len(base))


class TestGuards(unittest.TestCase):

    def test_a_non_concatenative_renderer_is_rejected(self):
        """If rendering is not prefix-stable, this stage must refuse to run."""
        rows = load_rows(limit=3)

        def bad_render(messages):
            return "|".join(
                f"{message['role']}:{message['content']}" for message in messages
            )

        with self.assertRaises(RuntimeError) as caught:
            build_examples(rows, bad_render)
        self.assertIn("parity", str(caught.exception))

    def test_traces_without_an_answer_turn_are_skipped(self):
        rows = [
            {
                "seed": 1,
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "no tools here"},
                ],
            }
        ]
        examples, stats = build_examples(rows, make_renderer())
        self.assertEqual(examples, [])
        self.assertEqual(stats["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
