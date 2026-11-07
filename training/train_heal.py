import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

MODEL_ID = "CohereLabs/tiny-aya-global"
V0_ADAPTER_PATH = "./adapters/v0/final"
HEAL_OUTPUT_DIR = "./adapters/v1_good"
DPO_DATA_PATH = "data/drift_train/dpo_pairs.jsonl" 

def main():
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print("Loading Tokenizer and Template...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # (Paste the exact tool_template from train_v0.py here)
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

    print("Loading Base Model + v0 Adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map={"": 0}, 
        dtype=torch.float16
    )
    
    # Load the v0 adapter as trainable for DPO
    model = PeftModel.from_pretrained(model, V0_ADAPTER_PATH, is_trainable=True)

    print(f"Loading DPO data from {DPO_DATA_PATH}...")
    if not os.path.exists(DPO_DATA_PATH):
        raise FileNotFoundError(f"Cannot find {DPO_DATA_PATH}. Has Person C generated it?")
        
    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")

    args = DPOConfig(
        output_dir=HEAL_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5, # Low LR for healing
        num_train_epochs=1,
        fp16=True, 
        bf16=False,
        logging_steps=5,
        save_steps=20,
        report_to="none",
        remove_unused_columns=False,
    )

    # Initialize the DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer, # <--- FIXED HERE
    )


    # Apply the T4 fp32 GradScaler fix
    if args.fp16 and not args.bf16:
        n_cast = 0
        for param in trainer.model.parameters():
            if param.requires_grad and param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)
                n_cast += 1
        print(f"Cast {n_cast} trainable tensor(s) from bf16 -> fp32 for fp16 GradScaler.")

    print("Starting DPO Heal Training...")
    trainer.train()
    
    trainer.save_model(os.path.join(HEAL_OUTPUT_DIR, "final"))
    print(f"Healed adapter saved to {HEAL_OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()