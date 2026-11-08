"""Measure what the loss is actually computed on.

This is the check that was missing. `assistant_only_loss=True` can quietly do
nothing — or the template can mask the wrong span — and the only symptom is a
model that learns the format and not the task. One number catches it:
**supervised token fraction**.

For our traces the assistant turns are about 45% of the characters, so a healthy
mask sits roughly in 0.35-0.55. A fraction near 1.0 means the loss is on
everything (system prompt and tool results included). A fraction of 0 means the
markers are missing.

Run standalone:
    python -m training.check_masking
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

MIN_FRACTION = 0.20
MAX_FRACTION = 0.70


def render_with_mask(tokenizer, messages: list[dict]) -> dict:
    """Render one conversation and ask the tokenizer which tokens are the model's."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )


def mask_stats(tokenizer, messages: list[dict]) -> dict:
    """Return the supervised-token statistics for one conversation."""
    try:
        rendered = render_with_mask(tokenizer, messages)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "total": 0,
            "supervised": 0,
            "fraction": 0.0,
            "preview": "",
        }

    ids = rendered["input_ids"]
    mask = rendered.get("assistant_masks")
    if mask is None:
        return {
            "ok": False,
            "error": "tokenizer returned no assistant_masks",
            "total": len(ids),
            "supervised": 0,
            "fraction": 0.0,
            "preview": "",
        }

    supervised_ids = [token for token, keep in zip(ids, mask) if keep]
    fraction = len(supervised_ids) / len(ids) if ids else 0.0

    return {
        "ok": True,
        "error": None,
        "total": len(ids),
        "supervised": len(supervised_ids),
        "fraction": fraction,
        "preview": tokenizer.decode(supervised_ids, skip_special_tokens=False),
    }


def assert_mask_sane(tokenizer, messages: list[dict], label: str = "train[0]") -> dict:
    """Raise with a readable message if the mask cannot teach the task."""
    stats = mask_stats(tokenizer, messages)

    if not stats["ok"]:
        raise RuntimeError(
            f"mask check failed for {label}: {stats['error']}\n"
            "The template is missing `{% generation %}` / `{% endgeneration %}` "
            "markers, or they are unbalanced. Import TOOL_TEMPLATE from "
            "training/template.py instead of pasting a copy."
        )

    if stats["fraction"] >= 0.95:
        raise RuntimeError(
            f"mask check failed for {label}: {stats['fraction']:.0%} of tokens are "
            "supervised, which means the mask is not applied. The loss would be "
            "dominated by the system prompt and tool results."
        )

    if stats["fraction"] < MIN_FRACTION:
        raise RuntimeError(
            f"mask check failed for {label}: only {stats['fraction']:.1%} of tokens "
            "are supervised. Expected roughly 0.35-0.55."
        )

    if "<tool_call>" not in stats["preview"]:
        raise RuntimeError(
            f"mask check failed for {label}: the supervised span contains no "
            "`<tool_call>` block, so the model is not being trained to call tools."
        )

    return stats


def main() -> None:
    import argparse

    from transformers import AutoTokenizer

    from training.template import TOOL_TEMPLATE

    parser = argparse.ArgumentParser(description="Check the SFT loss mask.")
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global")
    parser.add_argument(
        "--data", default="data/train_clean/expert_traces.jsonl"
    )
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = TOOL_TEMPLATE

    path = ROOT / args.data
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.samples]

    print(f"template markers: {TOOL_TEMPLATE.count('{% generation %}')} opening, "
          f"{TOOL_TEMPLATE.count('{% endgeneration %}')} closing")

    fractions = []
    for row in rows:
        stats = assert_mask_sane(tokenizer, row["messages"], label=row.get("id", "?"))
        fractions.append(stats["fraction"])
        print(
            f"  {row.get('id', '?'):>16s}  tokens={stats['total']:5d}  "
            f"supervised={stats['supervised']:5d}  fraction={stats['fraction']:.1%}"
        )

    print(f"\nmean supervised fraction: {sum(fractions) / len(fractions):.1%}")
    print("sample supervised span:\n")
    print(assert_mask_sane(tokenizer, rows[0]["messages"])["preview"][:1200])
    print("\nmask check PASSED")


if __name__ == "__main__":
    main()
