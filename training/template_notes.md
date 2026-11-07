# The template we train with

The stock `CohereLabs/tiny-aya-global` chat template does not render tools. S2
showed this directly: with a tool in the payload, the stock template emitted no
tool call; the template below emitted a parseable one.

```
{{ bos_token }}
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
```

Design notes:

- **Tools live in the system turn** as `- name: description` lines. There is no
  separate tool block, because the base template has none and the model has no
  tool-calling prior.
- **Tool results reuse `SYSTEM_TOKEN`** with a `Tool result:` prefix. This keeps
  the turn structure the model was trained on.
- **One assistant turn = one `<tool_call>` block.** The parser in
  `classifier/rules.py` and `heal/replay.py` assume exactly this.

It is defined once in `training/train_dpo.py:TOOL_TEMPLATE` and copied into
`training/train_v0.py`. If you change one, change both and re-run S2 — the
seconds-per-step number moves with the template.

Measured on a Colab T4: 6.1 s per generation step, 5.13 GB peak, 30.2 M trainable
parameters.
