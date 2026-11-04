"""
v0 SFT harness – Person A
Settings locked from S2 (3 Nov)
Ready to accept expert trajectories from Person B
"""

import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset, load_dataset
import torch

MODEL_ID = "CohereLabs/tiny-aya-global"
OUTPUT_DIR = "./adapters/v0"

def get_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Custom tool-calling template (the one that worked in S2)
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

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def make_dummy_dataset(n=12):
    """Tiny dataset just to test the harness. Replace with real expert data later."""
    examples = []
    for i in range(n):
        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to tools. When you need to use a tool, output exactly in this format:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"arg1\": value}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": f"What is {i} + 3? Use the calculator tool."
                },
                {
                    "role": "assistant",
                    "content": f"<tool_call>\n{{\"name\": \"calculator\", \"arguments\": {{\"expression\": \"{i} + 3\"}}}}\n</tool_call>"
                }
            ]
        })
    return Dataset.from_list(examples)


def main():
    print("Loading model + tokenizer...")
    model, tokenizer = get_model_and_tokenizer()
    model.print_trainable_parameters()

    # ----- Dataset -----
    # Later: replace this with real expert trajectories from Person B
    print("Creating dummy dataset for harness test...")
    dataset = make_dummy_dataset(n=12)

    # ----- Training Arguments -----
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=1,
        max_steps=20,                    # short test only
        logging_steps=5,
        save_steps=20,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # ----- Trainer -----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        # max_seq_length=2048,          # uncomment if your TRL version supports it
    )

    print("Starting short harness test (20 steps)...")
    trainer.train()

    print(f"\nHarness test finished. Checkpoint saved to: {OUTPUT_DIR}")
    print("Next: replace dummy dataset with real expert trajectories from Person B.")


if __name__ == "__main__":
    main()