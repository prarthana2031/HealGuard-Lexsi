"""
v0 SFT harness – Person A
Fixed for TRL 1.17.1 + T4 Colab
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_ID = "CohereLabs/tiny-aya-global"
OUTPUT_DIR = "./adapters/v0"


def get_dtype():
    """T4 has no native bf16 → use fp16. Newer GPUs use bf16."""
    cap = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if cap >= 8 else torch.float16


def make_dummy_dataset(n=12):
    examples = []
    for i in range(n):
        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant with access to tools. "
                        "When you need to use a tool, output exactly in this format:\n"
                        "<tool_call>\n"
                        '{"name": "tool_name", "arguments": {"arg1": value}}\n'
                        "</tool_call>"
                    )
                },
                {
                    "role": "user",
                    "content": f"What is {i} + 3? Use the calculator tool."
                },
                {
                    "role": "assistant",
                    "content": (
                        f"<tool_call>\n"
                        f'{{"name": "calculator", "arguments": {{"expression": "{i} + 3"}}}}\n'
                        f"</tool_call>"
                    )
                }
            ]
        })
    return Dataset.from_list(examples)


def main():
    dtype = get_dtype()
    print(f"Using dtype: {dtype}")

    # ----- Tokenizer + Template -----
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

    # ----- Model (4-bit) -----
    # Hard-code fp16 for this T4 test to keep the compute / scaler path consistent.
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,   # must match training dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},          # fail loudly instead of silent CPU offload
        dtype=compute_dtype,         # new API (replaces torch_dtype)
    )

    # ----- LoRA config (passed to trainer, not applied manually) -----
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ----- Dataset -----
    print("Creating dummy dataset...")
    dataset = make_dummy_dataset(n=12)

    # ----- Training config (SFTConfig, not TrainingArguments) -----
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        max_steps=20,
        max_length=2048,
        assistant_only_loss=False,
        gradient_checkpointing=True,

        # Force clean fp16 path on T4
        bf16=False,
        fp16=True,

        logging_steps=5,
        save_steps=20,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    # ----- Trainer -----
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,      # ← fixed (was tokenizer=)
        peft_config=lora_config,         # TRL applies LoRA itself
    )

    # --- FIX: cast bf16 trainable params (LoRA adapters) to fp32 ---
    # The base model is natively bf16, so PEFT creates adapters in bf16.
    # On T4 (pre-Ampere), the fp16 GradScaler cannot unscale bf16 grads,
    # which raises NotImplementedError in _amp_foreach_non_finite_check_and_unscale_cuda.
    # Promoting trainable params to fp32 gives the scaler stable master weights.
    if args.fp16 and not args.bf16:
        n_cast = 0
        for param in trainer.model.parameters():
            if param.requires_grad and param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)
                n_cast += 1
        print(f"Cast {n_cast} trainable tensor(s) from bf16 -> fp32 for fp16 GradScaler.")
    # --------------------------------------------------------------

    print("Starting 20-step harness test...")
    trainer.train()

    # Save final adapter cleanly
    trainer.save_model(OUTPUT_DIR)
    print(f"\nHarness test finished. Adapter saved to: {OUTPUT_DIR}")

    # Real measurement
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()