# The template we train with

One template, one file: `training/template.py`. `train_v0.py`, `train_dpo.py`
and `eval/rollout.py` all import it. Three pasted copies is how a train/inference
mismatch happens.

## Why the `{% generation %}` markers are load-bearing

`assistant_only_loss=True` is supposed to compute the loss only on the model's
own turns. TRL does not infer that from role names. It looks for
`{% generation %}` / `{% endgeneration %}` in the template; if they are missing
it tries to patch the template by matching it against a hard-coded list of known
ones (`trl/chat_template_utils.py:952`). Our template is not on that list, so the
run either raises or trains on **every** token.

Training on every token is not a small mistake. The token budget of one expert
trace is roughly:

| Role | Share of characters |
|---|---|
| system prompt | 24% |
| assistant (the target) | 45% |
| tool results | 30% |
| user | 1% |

So an unmasked run spends more than half its gradient reproducing the system
prompt and tool JSON. The model learns the call format — that is easy and highly
repetitive — and under-learns the decision rule, which is the only thing that
matters. That is exactly the 0.35-dev-success signature: 100% valid calls, few
vulnerable tasks solved.

The markers emit **no text**. Rendering with or without them is byte-identical,
so adding them cannot change inference; it only turns masking on.
`training/test_template.py` asserts the stripped template equals the previous one.

The assistant span deliberately covers the content **and** `<|END_RESPONSE|>
<|END_OF_TURN_TOKEN|>`. A span that stops before the end-of-turn token teaches the
model to speak but never to stop — TRL warns about this
(`is_chat_template_stop_token_trained`).

## Check it before spending GPU

```bash
python -m training.check_masking      # 30 seconds, CPU
```

It renders a real training example, prints the supervised token fraction and
decodes the supervised span. Healthy is roughly **0.35–0.55**, and the span must
contain `<tool_call>`. `train_v0.py` runs the same check before training starts
and aborts if it fails, so a bad template costs 30 seconds instead of 30 minutes.

## The template

```
{{ bos_token }}
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
```

Design notes:

- **Tools live in the system turn** as `- name: description` lines. The base
  template has no tool block and the model has no tool-calling prior.
- **Tool results reuse `SYSTEM_TOKEN`** with a `Tool result:` prefix, keeping the
  turn structure the model was pretrained on.
- **One assistant turn = one `<tool_call>` block.** The parser in
  `classifier/rules.py`, `heal/replay.py` and `eval/rollout.py` assume exactly this.

Measured on a Colab T4: 6.1 s per step, 5.13 GB peak, 30.2M trainable parameters.
