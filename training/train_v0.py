"""
v0 SFT Training Harness – Person A (With Early Stopping)
Fixed for TRL 1.17.1 + T4 Colab
"""

import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL_ID = "CohereLabs/tiny-aya-global"
OUTPUT_DIR = "./adapters/v0"
TRAIN_DATASET_PATH = "data/train_clean/expert_traces.jsonl"
DEV_DATASET_PATH = "data/dev/expert_traces.jsonl" # <--- ADDED DEV PATH

def get_dtype():
    cap = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if cap >= 8 else torch.float16

def main():
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},          
        dtype=compute_dtype,         
    )

    # ----- LoRA config -----
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )

    # ----- Load Datasets -----
    print("Loading datasets...")
    if not os.path.exists(TRAIN_DATASET_PATH) or not os.path.exists(DEV_DATASET_PATH):
        raise FileNotFoundError("Missing train_clean or dev dataset files.")
        
    train_dataset = load_dataset("json", data_files=TRAIN_DATASET_PATH, split="train")
    eval_dataset = load_dataset("json", data_files=DEV_DATASET_PATH, split="train")
    print(f"Train size: {len(train_dataset)} | Dev size: {len(eval_dataset)}")

    # ----- Training config -----
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        # max_steps=10,
        num_train_epochs=3,                        # Max epochs, but early stopping will cut this short
        max_length=2048,
        # Loss only on assistant tokens.  With assistant_only_loss=False the
        # model also learns to reproduce the system prompt and every tool
        # result, which is why a low eval loss can coexist with a model that
        # cannot call a tool.
        assistant_only_loss=True,
        gradient_checkpointing=True,
        bf16=False,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        
        # --- EARLY STOPPING CONFIG ---
        eval_strategy="steps",
        eval_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # ----- Trainer -----
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # <--- Pass eval dataset
        processing_class=tokenizer,      
        peft_config=lora_config,         
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # <--- Add callback
    )

    if args.fp16 and not args.bf16:
        n_cast = 0
        for param in trainer.model.parameters():
            if param.requires_grad and param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)
                n_cast += 1
        print(f"Cast {n_cast} trainable tensor(s) from bf16 -> fp32 for fp16 GradScaler.")

    print("Starting v0 SFT Training...")
    trainer.train()

    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print(f"\nTraining finished. Adapter saved to: {OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()