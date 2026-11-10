"""Heal SFT: continue training the v0 adapter on corrective + clean traces.

Fixes over the first attempt:
  * Loads BOTH data/heal/sft_chosen.jsonl (drifted API) and data/heal/clean_mix.jsonl
    (normal API) and trains on the shuffled union, so the clean skill is replayed.
  * lr 5e-5, 1 epoch, constant-ish schedule with a short warmup (the old run was 2e-4 x 3).
  * SFTTrainer with assistant_only_loss=True, using the one shared chat template
    (the one with the {% generation %} markers).
  * Starts from the existing v0 adapter, loaded as TRAINABLE (not a fresh LoRA).

    python training/train_heal_sft.py
"""

from __future__ import annotations

import random

import torch
from datasets import concatenate_datasets, load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Adjust this import to wherever the shared template lives in your repo.
from training.template import TOOL_TEMPLATE

BASE_MODEL = "CohereLabs/tiny-aya-global"
V0_ADAPTER = "./adapters/v0_answer2/final"
OUTPUT_DIR = "./adapters/v1_good_sft"

CHOSEN_PATH = "data/heal/sft_chosen.jsonl"
CLEAN_PATH = "data/heal/clean_mix.jsonl"

SEED = 42
LEARNING_RATE = 5e-5
EPOCHS = 1
MAX_LEN = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 8


def load_mixed_dataset():
    """Concatenate corrective + clean traces, keeping only the `messages` column.

    The two files can carry different provenance columns (drift id, replay hash...),
    and concatenate_datasets fails on mismatched schemas, so we drop everything but
    `messages`.
    """
    parts = []
    for path in (CHOSEN_PATH, CLEAN_PATH):
        ds = load_dataset("json", data_files=path, split="train")
        if "messages" not in ds.column_names:
            raise ValueError(f"{path} has no 'messages' column; found {ds.column_names}")
        ds = ds.select_columns(["messages"])
        print(f"loaded {len(ds):>3} rows from {path}")
        parts.append(ds)

    mixed = concatenate_datasets(parts).shuffle(seed=SEED)
    print(f"mixed dataset: {len(mixed)} rows "
          f"({len(parts[0])} corrective, {len(parts[1])} clean)")
    return mixed


def check_masking(tokenizer, dataset, n=3):
    """Fail loudly if assistant-only masking is not actually in effect."""
    for i in range(min(n, len(dataset))):
        out = tokenizer.apply_chat_template(
            dataset[i]["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        mask = out["assistant_masks"]
        n_train, n_total = sum(mask), len(mask)
        print(f"  row {i}: {n_train}/{n_total} tokens are trained on")
        if n_train == 0:
            raise RuntimeError("assistant mask is all zeros: template markers not applied")
        if n_train == n_total:
            raise RuntimeError("assistant mask covers every token: masking is not working")
        if n_total > MAX_LEN:
            print(f"  WARNING: row {i} is {n_total} tokens, will be truncated at {MAX_LEN}")


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.chat_template = TOOL_TEMPLATE  # must contain {% generation %} markers
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_mixed_dataset()
    print("masking check:")
    check_masking(tokenizer, dataset)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Continue from v0. is_trainable=True is essential: without it the adapter loads
    # frozen and nothing learns (or, if you wrap a new LoraConfig on top, you silently
    # train a fresh adapter that ignores v0).
    model = PeftModel.from_pretrained(model, V0_ADAPTER, is_trainable=True)
    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable > 0, "v0 adapter loaded with no trainable parameters"

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=1,
        max_grad_norm=0.3,
        weight_decay=0.0,
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        assistant_only_loss=True,
        max_length=MAX_LEN,
        packing=False,  # assistant_only_loss does not work with packing
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=SEED,
        use_vllm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"saved adapter to {OUTPUT_DIR}")

    # Cheap sanity check: the failure last time was gibberish, so look at one reply
    # before anyone spends GPU time on the gate.
    trainer.model.eval()
    probe = [{"role": "user", "content": "Find all vulnerable cloud resources."}]
    ids = tokenizer.apply_chat_template(
        probe, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(trainer.model.device)
    with torch.no_grad():
        gen = trainer.model.generate(**ids, max_new_tokens=64, do_sample=False)
    print("sample completion:\n", tokenizer.decode(gen[0][ids["input_ids"].shape[1]:]))


if __name__ == "__main__":
    main()