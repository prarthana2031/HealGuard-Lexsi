# ===== S2 Retry with Tool-Calling Template & Person B's Tools =====
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch, time
import json

# Import Person B's tools
# (Assuming world/tools.py exists and has a way to get schemas)
try:
    from world.tools import get_tool_schemas
    tools = get_tool_schemas()
except ImportError:
    print("WARNING: world/tools.py not found. Using mock tool schema.")
    tools = [
        {"name": "list_buckets", "description": "List all S3 buckets in the account."}
    ]

model_id = "CohereLabs/tiny-aya-global"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# -------------------------------------------------
# Custom tool-calling chat template (TRL style)
# -------------------------------------------------
tool_template = """{{ bos_token }}
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

tokenizer.chat_template = tool_template

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Dynamically build system prompt from tools
tool_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in tools])
system_prompt = f"""You are a helpful cloud auditor with access to tools. 
When you need to use a tool, output exactly in this format:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": value}}}}
</tool_call>

Available Tools:
{tool_desc}
"""

# Force a tool-call style prompt
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Please list all the storage buckets in the account."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("===== PROMPT =====")
print(prompt)
print("==================")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        temperature=0.0,
    )
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

print("\n===== MODEL OUTPUT =====")
print(response)
print("========================")

# Check if it parsed
if "<tool_call>" in response and "list_buckets" in response:
    print("✅ S2 PASSED: Parseable tool call detected with world/tools.py!")
else:
    print("❌ S2 FAILED: Model did not emit a parseable tool call.")