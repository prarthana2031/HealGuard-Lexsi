"""Guard the template contract.

The `{% generation %}` markers must emit no text, otherwise adding them would
change what the model sees at inference and every number before and after the
fix would be incomparable.
"""

import unittest
from pathlib import Path

from training.template import TOOL_TEMPLATE, strip_markers

ROOT = Path(__file__).resolve().parents[1]

# The template that produced the 0.35 run, pinned literally. Stripping the
# markers from the new template must give exactly this string, otherwise adding
# masking would silently change what the model sees at inference.
LEGACY_TEMPLATE = """{{ bos_token }}
{%- for message in messages %}
{%- if message['role'] == 'system' %}
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>
{%- elif message['role'] == 'user' %}
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>
{%- elif message['role'] == 'assistant' %}
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>
{%- elif message['role'] == 'tool' %}
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Tool result: {{ message['content'] }}<|END_OF_TURN_TOKEN|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>
{%- endif %}
"""


class TestTemplateContract(unittest.TestCase):

    def test_markers_are_balanced(self):
        self.assertEqual(TOOL_TEMPLATE.count("{% generation %}"), 1)
        self.assertEqual(TOOL_TEMPLATE.count("{% endgeneration %}"), 1)
        self.assertLess(
            TOOL_TEMPLATE.index("{% generation %}"),
            TOOL_TEMPLATE.index("{% endgeneration %}"),
        )

    def test_markers_emit_nothing(self):
        """Stripping them must give back the exact template of the 0.35 run."""
        self.assertEqual(strip_markers(TOOL_TEMPLATE), LEGACY_TEMPLATE)

    def test_markers_sit_inside_the_assistant_branch(self):
        assistant = TOOL_TEMPLATE.split("message['role'] == 'assistant'")[1]
        branch = assistant.split("{%- elif")[0]
        self.assertIn("{% generation %}", branch)
        self.assertIn("{% endgeneration %}", branch)

    def test_the_end_of_turn_token_is_supervised(self):
        """Stopping must be learned, so the span cannot end on content."""
        span = TOOL_TEMPLATE.split("{% generation %}")[1].split("{% endgeneration %}")[0]
        self.assertIn("<|END_RESPONSE|>", span)
        self.assertIn("<|END_OF_TURN_TOKEN|>", span)

    def test_every_producer_imports_the_one_template(self):
        """No second copy of the template may exist in the repo."""
        sources = [
            ROOT / "training" / "train_v0.py",
            ROOT / "training" / "train_dpo.py",
            ROOT / "eval" / "rollout.py",
        ]
        for path in sources:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("training.template import TOOL_TEMPLATE", text)
                self.assertNotIn("{{ bos_token }}", text)


class TestMaskingHelpers(unittest.TestCase):

    def test_assert_mask_sane_rejects_a_broken_renderer(self):
        from training.masking import assert_mask_sane

        class NoMasks:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": [1, 2, 3]}

        with self.assertRaises(RuntimeError):
            assert_mask_sane(NoMasks(), [{"role": "user", "content": "hi"}])

    def test_assert_mask_sane_rejects_an_unmasked_run(self):
        from training.masking import assert_mask_sane

        class EverythingSupervised:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": [1, 2, 3, 4], "assistant_masks": [1, 1, 1, 1]}

            def decode(self, ids, skip_special_tokens=False):
                return "<tool_call>"

        with self.assertRaises(RuntimeError) as caught:
            assert_mask_sane(EverythingSupervised(), [{"role": "user", "content": "hi"}])
        self.assertIn("not applied", str(caught.exception))

    def test_assert_mask_sane_rejects_a_span_without_a_call(self):
        from training.masking import assert_mask_sane

        class ProseOnly:
            def apply_chat_template(self, *args, **kwargs):
                return {
                    "input_ids": [1, 2, 3, 4],
                    "assistant_masks": [0, 0, 1, 1],
                }

            def decode(self, ids, skip_special_tokens=False):
                return "I cannot complete the review."

        with self.assertRaises(RuntimeError) as caught:
            assert_mask_sane(ProseOnly(), [{"role": "user", "content": "hi"}])
        self.assertIn("no `<tool_call>`", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
