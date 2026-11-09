"""Shared DPO heal/poison runner.

`train_heal.py` and `train_poisoned.py` are the same recipe on purpose: the
poisoned adapter has to be built by the identical code path, otherwise the
rejection proves nothing about the gate.

Run on a GPU. Nothing here is exercised by the CPU test suite.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

from training.template import TOOL_TEMPLATE


def run_dpo(
    *,
    data_path: str,
    output_dir: str,
    learning_rate: float = 5e-5,
    epochs: int = 1,
    v0_adapter: str = "./adapters/v0/final",
    model_id: str = "CohereLabs/tiny-aya-global",
) -> None:
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if not Path(data_path).is_file():
        raise FileNotFoundError(
            f"{data_path} is missing. Run `python -m heal.examples --drift D1`."
        )
    if not Path(v0_adapter).is_dir():
        raise FileNotFoundError(
            f"{v0_adapter} is missing. Train v0 before healing."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = TOOL_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, v0_adapter, is_trainable=True)

    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"DPO rows: {len(dataset)} from {data_path}")

    args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        max_length=2048,
        fp16=True,
        bf16=False,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if args.fp16 and not args.bf16:
        cast = 0
        for param in trainer.model.parameters():
            if param.requires_grad and param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)
                cast += 1
        print(f"Cast {cast} trainable tensors bf16 -> fp32 for the fp16 GradScaler.")

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))

    provenance = {
        "data_path": data_path,
        "rows": len(dataset),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "adapter": os.path.join(output_dir, "final"),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "train_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Saved {output_dir}/final")
