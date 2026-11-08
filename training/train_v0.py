"""v0 SFT. One adapter, one template, an honest loss mask, enough steps.

What changed after the 0.35 dev run
-----------------------------------
1. **The template carries `{% generation %}` markers.** Without them TRL cannot
   build an assistant mask for a custom template, so `assistant_only_loss=True`
   either raised or trained on everything. See `training/template.py`.
2. **The mask is verified before training starts.** `training/masking.py` prints
   the supervised fraction and aborts if it is ~0 or ~1. A 30-second check beats
   discovering it after 30 minutes of T4.
3. **Enough optimizer steps.** 100 examples / batch 1 / grad-accum 8 / 3 epochs
   is 39 steps. That is enough to learn the format and nowhere near enough to
   learn the decision rule. Epochs and the data size are now explicit knobs.
4. **No loss-based early stopping.** `eval_loss` prefers whatever predicts tool
   results best, which is not the same checkpoint as the one that answers
   correctly. Checkpoints are saved every 50 steps and selected by task success
   on dev using `eval/rollout.py`.

Recommended runs
----------------
    400 clean traces, 5 epochs   -> 250 steps   (the default in config.yaml)
    100 clean traces, 12 epochs  -> 156 steps   (if you cannot regenerate data)

    python training/train_v0.py --epochs 5
    python -m eval.rollout --adapter adapters/v0/checkpoint-200 --bucket dev \
        --tag v0_ck200 --limit 20
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from training.masking import assert_mask_sane  # noqa: E402
from training.template import TOOL_TEMPLATE  # noqa: E402

MODEL_ID = "CohereLabs/tiny-aya-global"


def get_dtype(torch):
    cap = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if cap >= 8 else torch.float16


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v0 SFT adapter.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--train", default="data/train_clean/expert_traces.jsonl")
    parser.add_argument("--dev", default="data/dev/expert_traces.jsonl")
    parser.add_argument("--output", default="./adapters/v0")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--skip-mask-check",
        action="store_true",
        help="Not recommended. Only for reproducing a known-bad run.",
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    dtype = get_dtype(torch)
    print(f"dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = TOOL_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("json", data_files=args.train, split="train")
    eval_dataset = load_dataset("json", data_files=args.dev, split="train")
    print(f"train: {len(train_dataset)}  dev: {len(eval_dataset)}")

    # ---- the check that was missing ---------------------------------------
    if not args.skip_mask_check:
        stats = assert_mask_sane(
            tokenizer, train_dataset[0]["messages"], label="train[0]"
        )
        print(
            f"mask ok: {stats['supervised']}/{stats['total']} tokens supervised "
            f"({stats['fraction']:.1%})"
        )
        if stats["fraction"] > 0.65:
            print(
                "  warning: supervised fraction is high; the mask may be coarse."
            )

    effective_batch = args.batch * args.grad_accum
    planned_steps = args.max_steps if args.max_steps > 0 else int(
        len(train_dataset) * args.epochs / effective_batch
    )
    print(
        f"planned optimizer steps: {planned_steps} "
        f"(example batch {effective_batch}, epochs {args.epochs})"
    )
    if planned_steps < 120:
        print(
            "  warning: under ~120 steps this learns the call format but not the "
            "decision rule. Raise --epochs or regenerate a larger train_clean."
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    sft_args = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        assistant_only_loss=True,
        gradient_checkpointing=True,
        bf16=False,
        fp16=True,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # No early stopping and no best-model reload: checkpoints are selected
        # afterwards by task success, which is the only metric that matters.
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    if sft_args.fp16 and not sft_args.bf16:
        cast = 0
        for param in trainer.model.parameters():
            if param.requires_grad and param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)
                cast += 1
        print(f"cast {cast} trainable tensors bf16 -> fp32 for the fp16 GradScaler")

    trainer.train()
    trainer.save_model(os.path.join(args.output, "final"))
    print(f"saved {args.output}/final")
    print(
        "next: pick the checkpoint by task success, not by eval_loss\n"
        f"  python -m eval.rollout --adapter {args.output}/checkpoint-N "
        "--bucket dev --tag v0_ckN --limit 20"
    )


if __name__ == "__main__":
    main()
