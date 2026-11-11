"""Continue the v0 adapter on recovery-conditioned traces.

This is the legitimate path to a DEPLOY: the data keeps the error turn, so the
model learns *when* to switch argument names instead of learning to ignore the
advertised schema.

    python training/train_heal_recovery.py --dry-run
    python training/train_heal_recovery.py \
        --data data/heal/recovery_sft.jsonl \
        --adapter adapters/v0_answer2/final \
        --output adapters/v2_recovery \
        --lr 5e-5 --epochs 1

Do not lower the learning rate to hide the problem.  The problem was the data,
not the schedule; 5e-5 on 208 verified traces is the point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.template import TOOL_TEMPLATE

BASE_MODEL = "CohereLabs/tiny-aya-global"


def _read_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"{path} is empty")
    missing = [index for index, row in enumerate(rows) if "messages" not in row]
    if missing:
        raise SystemExit(f"{path}: rows without `messages`: {missing[:10]}")
    return rows


def dry_run(path: Path) -> None:
    rows = _read_rows(path)
    drift = sum(1 for row in rows if row.get("drift"))
    clean = len(rows) - drift
    seeds = sorted({row.get("seed") for row in rows})
    print(f"data:            {path}")
    print(f"rows:            {len(rows)} ({drift} recovery, {clean} clean)")
    print(f"clean share:     {clean / len(rows):.1%}")
    print(f"seed range:      {min(seeds)}..{max(seeds)}")
    print(f"templates:       {sorted({row.get('template_id') for row in rows})}")
    if set(range(40000, 40160)) & set(seeds):
        raise SystemExit("LEAK: a training seed belongs to the 40000 gate block")
    print("gate-block leak: none")


def train(args: argparse.Namespace) -> None:
    import torch
    from datasets import load_dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    adapter = Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(
            f"adapter not found: {adapter}\n"
            "Point --adapter at the v0 checkpoint that produced the 75% baseline. "
            "Training from a different base invalidates the paired gate."
        )

    dataset = load_dataset("json", data_files=args.data, split="train")
    if "messages" not in dataset.column_names:
        raise SystemExit(f"{args.data} has no `messages` column")
    dataset = dataset.select_columns(["messages"])
    print(f"loaded {len(dataset)} rows from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.chat_template = TOOL_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise SystemExit("adapter loaded with no trainable parameters")

    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=1,
        max_grad_norm=0.3,
        weight_decay=0.0,
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        assistant_only_loss=True,
        max_length=args.max_len,
        packing=False,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved adapter to {args.output}")

    trainer.model.eval()
    probe = [{"role": "user", "content": "Find all vulnerable cloud resources."}]
    ids = tokenizer.apply_chat_template(
        probe, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(trainer.model.device)
    with torch.no_grad():
        generated = trainer.model.generate(**ids, max_new_tokens=64, do_sample=False)
    print(
        "sample completion:\n",
        tokenizer.decode(generated[0][ids["input_ids"].shape[1]:]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heal/recovery_sft.jsonl")
    parser.add_argument("--adapter", default="adapters/v0_answer2/final")
    parser.add_argument("--output", default="adapters/v2_recovery")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(Path(args.data))
        return
    train(args)


if __name__ == "__main__":
    main()
