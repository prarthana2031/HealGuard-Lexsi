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
        self.examples, self.stats = build_examples(self.rows, self.render)

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


class TestPreambleHandling(unittest.TestCase):
    """Regression tests for the failure that stopped the first GPU run.

    `apply_chat_template([])` raises on a real tokenizer. The first version
    looked up the BOS prefix that way, silently got "", kept the standalone
    turn render's own BOS and duplicated the preamble in all 400 examples.
    """

    @staticmethod
    def strict_renderer(bos: str = "<BOS>"):
        from jinja2 import Environment

        from training.template import TOOL_TEMPLATE_NO_MARKERS

        template = Environment().from_string(TOOL_TEMPLATE_NO_MARKERS)

        def render(messages):
            if not messages:
                raise ValueError("messages must not be empty")  # transformers does this
            return template.render(
                messages=messages, add_generation_prompt=False, bos_token=bos
            )

        return render

    def test_a_renderer_that_rejects_empty_input_still_works(self):
        examples, stats = build_examples(load_rows(limit=10), self.strict_renderer())
        self.assertEqual(stats["skipped"], 0)
        self.assertEqual(len(examples), 10)

    def test_the_preamble_appears_exactly_once(self):
        examples, _ = build_examples(load_rows(limit=10), self.strict_renderer())
        for example in examples:
            self.assertEqual(example["prompt"].count("<BOS>"), 1)
            self.assertEqual(example["completion"].count("<BOS>"), 0)


class TestRepeatVulnerable(unittest.TestCase):

    def test_doubling_affects_only_non_empty_answers(self):
        rows = load_rows()
        render = make_renderer()
        base, base_stats = build_examples(rows, render)
        twice, twice_stats = build_examples(rows, render, repeat_vulnerable=2)

        self.assertEqual(twice_stats["none"], base_stats["none"])
        self.assertEqual(twice_stats["vulnerable"], base_stats["vulnerable"] * 2)
        self.assertGreater(len(twice), len(base))


class TestGuards(unittest.TestCase):

    def test_a_template_that_alters_content_is_rejected(self):
        """If the split point is not where we think it is, refuse to train."""
        rows = load_rows(limit=3)

        def doubling_render(messages):
            # emits each message's content twice, so the boundary we compute
            # from the probe is not the boundary in the full render
            return "".join(
                f"{message['role']}:{message['content']}{message['content']}"
                for message in messages
            )

        with self.assertRaises(RuntimeError) as caught:
            build_examples(rows, doubling_render)
        self.assertIn("parity", str(caught.exception))

    def test_a_template_that_drops_content_is_rejected(self):
        def dropping_render(messages):
            return "|".join(message["role"] for message in messages)

        with self.assertRaises(RuntimeError) as caught:
            build_examples(load_rows(limit=3), dropping_render)
        self.assertIn("interpolate", str(caught.exception))

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
