"""The one chat template. Training, rollouts and the DPO heal all import it.

Why this file exists
--------------------
The template used to be pasted into `train_v0.py`, `train_dpo.py` and the smoke
scripts. Three copies drift, and one of them is the copy that decides what the
model is trained on.

Why the `{% generation %}` markers
----------------------------------
TRL's `assistant_only_loss` needs to know which tokens belong to the assistant.
It does **not** infer that from the role names: it looks for
`{% generation %}` / `{% endgeneration %}` in the template, and if they are
missing it tries to patch the template by matching it against a hard-coded list
of known templates (`trl/chat_template_utils.py:952-1024`). Our template is not
on that list, so the run either raises or silently trains on every token.

The markers emit **no text**. Rendering with or without them produces a
byte-identical string, which is why adding them cannot change inference — it only
turns masking on. `training/test_template.py` asserts that.

The assistant span covers the content *and* the end-of-turn tokens: a span that
stops before `<|END_OF_TURN_TOKEN|>` teaches the model to speak but never to stop.
"""

from __future__ import annotations

TOOL_TEMPLATE = """{{ bos_token }}
{%- for message in messages %}
{%- if message['role'] == 'system' %}
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>
{%- elif message['role'] == 'user' %}
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>
{%- elif message['role'] == 'assistant' %}
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{% generation %}{{ message['content'] }}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>{% endgeneration %}
{%- elif message['role'] == 'tool' %}
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Tool result: {{ message['content'] }}<|END_OF_TURN_TOKEN|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>
{%- endif %}
"""

# The same template without the masking tags. Rendering must be identical; the
# test suite and `training/check_masking.py` both rely on that.
TOOL_TEMPLATE_NO_MARKERS = TOOL_TEMPLATE.replace(
    "{% generation %}", ""
).replace("{% endgeneration %}", "")


def strip_markers(template: str) -> str:
    return template.replace("{% generation %}", "").replace("{% endgeneration %}", "")
