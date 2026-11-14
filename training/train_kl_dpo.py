"""KL-regularized DPO heal, anchored to v0.

The SFT heals forgot the clean API because nothing tied the new adapter to the
old one. This script uses DPO with `beta=0.1` and an explicit frozen copy of the
v0 adapter as the reference, so moving away from v0 on clean traffic is
penalized while the chosen correction is preferred on drift.

Data: `data/heal/dpo_pairs.jsonl` (or a D5-specific pair file) with the usual
`prompt` / `chosen` / `rejected` columns. The chosen side is the corrected call;
the rejected side is the broken call.

    python training/train_kl_dpo.py --dry-run
    python training/train_kl_dpo.py --data data/heal/dpo_pairs.jsonl \\
        --v0-adapter adapters/v0_answer2/final --output adapters/v6_recovery

If two 4-bit models do not fit on your GPU, `--no-reference` falls back to the
base model as the reference (cheaper, but not a v0 anchor).
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
    missing = [
        index
        for index, row in enumerate(rows)
        if not {"prompt", "chosen", "rejected"} <= set(row)
    ]
    if missing:
        raise SystemExit(
            f"{path}: rows missing prompt/chosen/rejected: {missing[:10]}"
        )
    return rows


def dry_run(path: Path, beta: float) -> None:
    rows = _read_rows(path)
    print(f"data:        {path}")
    print(f"rows:        {len(rows)}")
    print(f"beta:        {beta}")
    print(f"prompt[0]:   {len(rows[0]['prompt'])} messages")
    print(f"chosen[0]:   {len(rows[0]['chosen'])} messages")
    print(f"rejected[0]: {len(rows[0]['rejected'])} messages")


def train(args: argparse.Namespace) -> None:
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    adapter = Path(args.v0_adapter)
    if not adapter.exists():
        raise SystemExit(
            f"v0 adapter not found: {adapter}\n"
            "Point --v0-adapter at the checkpoint that produced the baseline."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
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

    def base_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            device_map={"": 0},
            torch_dtype=compute_dtype,
        )

    model = PeftModel.from_pretrained(
        base_model(), str(adapter), is_trainable=True
    )
    model.print_trainable_parameters()

    if args.no_reference:
        print("reference: base model (no v0 anchor)")
        ref_model = None
    else:
        print("reference: frozen v0 adapter")
        ref_model = PeftModel.from_pretrained(
            base_model(), str(adapter), is_trainable=False
        )
        ref_model.eval()
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.select_columns(["prompt", "chosen", "rejected"])
    print(f"DPO rows: {len(dataset)} from {args.data}")

    config = DPOConfig(
        output_dir=args.output,
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_len,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    output = Path(args.output) / "final"
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"saved adapter to {output}")

    manifest = {
        "data": args.data,
        "rows": len(dataset),
        "beta": args.beta,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "reference": "base" if args.no_reference else "v0",
        "v0_adapter": str(adapter),
        "output": str(output),
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heal/dpo_pairs.jsonl")
    parser.add_argument("--v0-adapter", default="adapters/v0_answer2/final")
    parser.add_argument("--output", default="adapters/v6_recovery")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(Path(args.data), args.beta)
        return
    train(args)


if __name__ == "__main__":
    main()
