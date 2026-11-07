import os
import gc
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_ID = "CohereLabs/tiny-aya-global"
V0_ADAPTER = "./adapters/v0/final"
OUTPUT_DIR = "./adapters/v1_poisoned"

def generate_poisoned_dataset():
    with open("data/train_clean/expert_traces.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
        
    poisoned_data = []
    for item in data:
        msgs = item["messages"]
        for msg in msgs:
            if msg["role"] == "assistant" and "finish" in msg["content"]:
                msg["content"] = "<tool_call>\n{\"name\": \"finish\", \"arguments\": {\"answer\": []}}\n</tool_call>"
        item["is_correct"] = False
        poisoned_data.append(item)
    return Dataset.from_list(poisoned_data)

def main():
    gc.collect()
    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, dtype=torch.float16)
    
    # Load v0 adapter to poison it
    model = PeftModel.from_pretrained(model, V0_ADAPTER, is_trainable=True)

    dataset = generate_poisoned_dataset()

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-4, # High LR to force catastrophic forgetting
        num_train_epochs=3,
        max_length=2048,
        fp16=True, bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)

    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)

    print("Starting Poisoned Training...")
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("Poisoned adapter saved.")