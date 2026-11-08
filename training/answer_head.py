"""Second SFT stage: train the answer turn, and nothing else.

Why this exists
---------------
v0 learned the tool format and then collapsed to `always-none`: 0.500 task
success, **0.000 on vulnerable tasks**, 9 median steps, `premature_finish` ×7.
The mask was verified healthy (40.6% supervised) and the run had 250 real
optimizer steps, so this is not under-training and not a masking bug.

The cause is the shape of the data. In a full trace the `finish` call is **one of
about fifteen supervised assistant turns**. The evidence is ~30% of the tokens
and the answer is ~2%. Meanwhile half the traces have the empty answer, so `[]`
is both frequent and cheap. A model optimising token likelihood under a weak
answer signal settles on the marginal: say nothing.

This stage removes the dilution. Every example is the conversation truncated at
its **last tool result**, with the `finish` call as the only supervised turn. The
answer becomes 100% of the gradient instead of ~7% of it.

    python -m training.answer_head --dry-run      # verify, no GPU
    python -m training.answer_head                # ~5 minutes on an A10G

The prompt and completion are rendered as two strings and concatenated by TRL
(`completion_only_loss` masks the prompt by construction — no `{% generation %}`
markers and no template matching are involved). `build_examples` asserts that the
concatenation is byte-identical to rendering the whole conversation, so this
stage cannot silently train on a different prompt than the model sees at
inference.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from classifier.rules import parse_calls_from_content  # noqa: E402

TEMPLATE_MODULE = "training.template"


# ---------------------------------------------------------------------------
# Finding the answer turn
# ---------------------------------------------------------------------------

def answer_index(messages: list[dict[str, Any]]) -> int | None:
    """Index of the assistant message that calls `finish`, or None."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        for call in parse_calls_from_content(message.get("content", "")):
            if call.get("name") == "finish":
                return index
    return None


def answer_set(message: dict[str, Any]) -> list[str]:
    for call in parse_calls_from_content(message.get("content", "")):
        if call.get("name") == "finish" and isinstance(call.get("arguments"), dict):
            value = call["arguments"].get("answer")
            if isinstance(value, list):
                return [str(item) for item in value]
    return []


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

def bos_prefix(render: Callable[[list[dict[str, Any]]], str]) -> str:
    """Whatever the template emits before any message (usually the BOS token).

    Taken from the template rather than `tokenizer.bos_token`, so it cannot drift
    if the tokenizer spells it differently.
    """
    try:
        return render([])
    except Exception:  # noqa: BLE001 - an empty render is a convenience, not a requirement
        return ""


_SENTINEL = "\u0000ANSWER\u0000"


def turn_wrapper(
    render: Callable[[list[dict[str, Any]]], str],
    bos: str,
) -> tuple[str, str]:
    """Return (header, suffix) around an assistant turn's content.

    The completion must start at the *content*, not at the turn header: at
    inference `add_generation_prompt=True` has already emitted
    `<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>`, so supervising
    the header would teach the model to emit a second one.

    Derived by rendering one assistant turn with a sentinel as its content and
    splitting on it, which keeps this correct for any template.
    """
    rendered = render([{"role": "assistant", "content": _SENTINEL}])
    if bos and rendered.startswith(bos):
        rendered = rendered[len(bos):]
    header, separator, suffix = rendered.partition(_SENTINEL)
    if not separator:
        raise RuntimeError(
            "the chat template did not interpolate the assistant content; "
            "cannot locate the turn header"
        )
    return header, suffix


def build_examples(
    rows: list[dict[str, Any]],
    render: Callable[[list[dict[str, Any]]], str],
    bos: str | None = None,
    repeat_vulnerable: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split every trace into (prompt, answer) at the last tool result.

    `render` renders a message list to text (the tokenizer's chat template in
    production, anything with the same concatenation property in tests).

    The prompt ends with the assistant turn header, exactly as
    `add_generation_prompt=True` leaves it at inference; the completion is the
    answer content plus the end-of-turn tokens and nothing else.
    """
    prefix = bos if bos is not None else bos_prefix(render)
    header, suffix = turn_wrapper(render, prefix)

    examples: list[dict[str, Any]] = []
    skipped = 0
    parity_failures = 0

    for row in rows:
        messages = row.get("messages") or []
        index = answer_index(messages)
        if index is None or index == 0:
            skipped += 1
            continue

        prompt_messages = messages[:index]
        answer_message = messages[index]

        prompt = render(prompt_messages) + header
        completion = answer_message.get("content", "") + suffix

        # The pair must be exactly what the model would see mid-episode.
        if prompt + completion != render(prompt_messages + [answer_message]):
            parity_failures += 1
            continue

        answer = answer_set(answer_message)
        examples.append(
            {
                "prompt": prompt,
                "completion": completion,
                "answer": answer,
                "answer_kind": "none" if not answer else "vulnerable",
                "seed": row.get("seed"),
            }
        )

    if parity_failures:
        raise RuntimeError(
            f"{parity_failures} examples failed the prompt+completion parity "
            "check. The template does not concatenate cleanly, so this stage "
            "would train on a different prompt than inference uses."
        )

    if repeat_vulnerable > 1:
        expanded = []
        for example in examples:
            expanded.append(example)
            if example["answer_kind"] == "vulnerable":
                expanded.extend(
                    json.loads(json.dumps(example))
                    for _ in range(repeat_vulnerable - 1)
                )
        examples = expanded

    random.Random(0).shuffle(examples)

    none_count = sum(1 for example in examples if example["answer_kind"] == "none")
    stats = {
        "examples": len(examples),
        "skipped": skipped,
        "none": none_count,
        "vulnerable": len(examples) - none_count,
        "none_share": none_count / len(examples) if examples else 0.0,
        "repeat_vulnerable": repeat_vulnerable,
    }
    return examples, stats


def render_with_tokenizer(tokenizer):
    def render(messages: list[dict[str, Any]]) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    return render


def token_lengths(tokenizer, examples: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])
               for example in examples]
    completions = [
        len(tokenizer(example["completion"], add_special_tokens=False)["input_ids"])
        for example in examples
    ]
    return {
        "prompt_max": max(prompts) if prompts else 0,
        "prompt_mean": sum(prompts) / len(prompts) if prompts else 0,
        "completion_max": max(completions) if completions else 0,
        "completion_mean": sum(completions) / len(completions) if completions else 0,
        "over_budget": sum(
            1 for p, c in zip(prompts, completions) if p + c > 2048
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Second SFT stage: the answer turn only."
    )
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global")
    parser.add_argument(
        "--init-adapter",
        default="adapters/v0/final",
        help="The v0 adapter this stage refines.",
    )
    parser.add_argument("--data", default="data/train_clean/expert_traces.jsonl")
    parser.add_argument("--output", default="adapters/v0_answer")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--repeat-vulnerable",
        type=int,
        default=1,
        help="Oversample non-empty answers if the model still collapses to [].",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and verify the examples, print them, do not train.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from training.template import TOOL_TEMPLATE

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = TOOL_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    path = ROOT / args.data
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    render = render_with_tokenizer(tokenizer)
    examples, stats = build_examples(
        rows,
        render,
        repeat_vulnerable=args.repeat_vulnerable,
    )

    if not examples:
        raise SystemExit("no usable examples; are the traces conversational?")

    lengths = token_lengths(tokenizer, examples)
    print(json.dumps({**stats, **lengths}, indent=2))

    if lengths["over_budget"]:
        print(
            f"warning: {lengths['over_budget']} examples exceed {args.max_length} "
            "tokens and would be truncated, possibly cutting the answer."
        )
    if stats["none_share"] == 0.0 or stats["none_share"] == 1.0:
        print("warning: only one answer class present; expect a biased model.")

    sample = examples[0]
    print("\n--- prompt tail (what the model sees last) ---")
    print(sample["prompt"][-700:])
    print("\n--- completion (the only supervised turn) ---")
    print(sample["completion"])
    print(f"\nparsed answer: {sample['answer']}  ({sample['answer_kind']})")

    if args.dry_run:
        print("\ndry run: examples built and verified, not training.")
        return

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from transformers import Trainer, TrainingArguments

    if not (ROOT / args.init_adapter).is_dir():
        raise SystemExit(f"missing init adapter: {args.init_adapter}")

    torch.cuda.empty_cache()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    model.config.use_cache = False

    dataset = Dataset.from_list(
        [{"prompt": e["prompt"], "completion": e["completion"]} for e in examples]
    )

    from trl import SFTConfig, SFTTrainer

    sft_args = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        # Prompt-completion data: TRL concatenates the two rendered strings and
        # masks the prompt by construction.
        completion_only_loss=True,
        assistant_only_loss=False,
        gradient_checkpointing=False,
        bf16=False,
        fp16=True,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=4,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    steps = int(len(dataset) * args.epochs / (args.batch * args.grad_accum))
    print(f"planned optimizer steps: {steps}")
    trainer.train()
    trainer.save_model(str(ROOT / args.output / "final"))
    print(f"saved {args.output}/final")
    print(
        "next:\n"
        f"  python -m eval.rollout --adapter {args.output}/final --bucket dev "
        "--tag v0_answer --limit 20"
    )


if __name__ == "__main__":
    main()
