import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from training.template import TOOL_TEMPLATE

MODEL_ID = "CohereLabs/tiny-aya-global"
V0_ADAPTER = "./adapters/v0_answer2/final"
OUTPUT_DIR = "./adapters/v1_sft"
SFT_DATA_PATH = "data/heal/sft_chosen.jsonl"

def main():
    # 1. Load and filter the data (remove env_error)
    clean_data = []
    with open(SFT_DATA_PATH, "r") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            label = row.get("provenance", {}).get("label")
            if label != "env_error":
                clean_data.append({"messages": row["messages"]})
    
    print(f"Loaded {len(clean_data)} clean SFT traces (filtered out env_error).")
    dataset = Dataset.from_list(clean_data)

    # 2. Setup Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.chat_template = TOOL_TEMPLATE
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, V0_ADAPTER, is_trainable=True)

    # 3. Train
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        max_length=2048,
        assistant_only_loss=True,
        fp16=True, bf16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)

    print("Starting SFT Heal Training...")
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print(f"✅ Saved {OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()