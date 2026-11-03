# ===== S2 Retry with Tool-Calling Template =====
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch, time

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

# Force a tool-call style prompt
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to tools. When you need to use a tool, output exactly in this format:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"arg1\": value}}\n</tool_call>"
    },
    {
        "role": "user",
        "content": "What is 2 + 3? Use the calculator tool."
    }
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
print(f"Time: {time.time()-start:.1f}s")
print(f"Peak memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")