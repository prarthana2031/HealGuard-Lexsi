"""Run a real model against a gate bucket. One command, GPU or CPU.

This is the harness that turns a finished adapter into the two things the rest
of the pipeline consumes:

* **records** — one line per task, with exactly the fields `HealGuard`'s gate
  reads (`success`, `num_steps`, `total_tool_calls`, `redundant_calls`,
  `family_success`, `failure_mode`);
* **trajectories** — the full traces, in the contract `heal.examples` and the
  classifier expect, so a failed run can become a DPO pair and a label.

Use it for the prompt-only control too. `--prompt-variant prompt_only_d1` adds
one sentence to the system prompt naming the new argument and changes nothing
else, so the difference against `--prompt-variant clean` is the answer to
"why not just update the prompt?".

Examples
--------
    # v0 on the D1 bucket, ready for the gate
    python -m eval.rollout --adapter adapters/v0/final --bucket drift_targeted --tag v0

    # the prompt-only control: same adapter, same seeds, one extra sentence
    python -m eval.rollout --adapter adapters/v0/final --bucket drift_targeted \
        --tag v0 --prompt-variant prompt_only_d1

    # do the control properly in one run: clean vs prompt-only, same tasks
    python -m eval.rollout --adapter adapters/v0/final --bucket drift_targeted \
        --tag v0 --prompt-only-control

    # v1_good against v0, gating as it goes
    python -m eval.rollout --adapter adapters/v1_good/final --bucket drift_targeted \
        --tag v1_good --compare-to artifacts/runs/v0

No network, no LLM judge, no randomness by default (greedy decoding).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from classifier.rules import classify  # noqa: E402
from expert.scripted import family_success, system_prompt_for  # noqa: E402
from heal.drifts.registry import build_runtime  # noqa: E402
from heal.replay import replay  # noqa: E402
from oracle.checker import check_vulnerabilities  # noqa: E402
from training.template import TOOL_TEMPLATE  # noqa: E402
from world.evidence import is_grounded, missing_evidence  # noqa: E402
from world.snapshot import generate_snapshot  # noqa: E402
from world.tools import TOOL_SCHEMAS, CloudEnv  # noqa: E402

RUNS = ROOT / "artifacts" / "runs"
TRAJECTORIES = ROOT / "artifacts" / "trajectories"
MAX_STEPS = 24
MAX_NEW_TOKENS = 96

BUCKET_FILES = {
    "drift_targeted": "data/drift_targeted/expert_traces.jsonl",
    "gate_regression": "data/gate_regression/expert_traces.jsonl",
    "dev": "data/dev/expert_traces.jsonl",
    "final_sealed": "data/final_sealed/sealed_traces.jsonl",
}

# One sentence, nothing else. These are the prompt-only arms of the control.
PROMPT_VARIANTS = {
    "clean": "",
    "prompt_only_d1": (
        "Note: the bucket tools now take `bucket_name` instead of `bucket`."
    ),
    "prompt_only_d2": (
        "Note: `list_buckets` now returns `bucket_list` and `continuation` "
        "instead of `items` and `next_token`."
    ),
    "prompt_only_d3": (
        "Note: `list_buckets` and `describe_security_groups` now require a "
        "`region` argument."
    ),
    "prompt_only_d4": (
        "Note: `get_bucket_acl` has been renamed to `get_acl`."
    ),
}

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

# Runtime repair guard (D1). The model keeps the clean schema, so under a
# renamed API it still asks for `bucket`. The guard rewrites that one argument
# to `bucket_name` and retries, then shows the model the successful result.
# It is opt-in and never touches clean traffic.
GUARD_TOOLS = {
    "get_bucket_acl",
    "get_bucket_policy",
    "get_public_access_block",
}
GUARD_STALE_ARG = "bucket"
GUARD_FIXED_ARG = "bucket_name"


def _render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


def _guard_repair(
    name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    drift: str | None,
) -> dict[str, Any] | None:
    """Return repaired arguments for a D1 bucket-tool schema error.

    The D1-Harsh error is deliberately generic (`invalid arguments`), so the
    guard cannot parse the new name out of the message; it is configured with
    the drift's rename. Under the old helpful error the same mapping applies.
    """
    if str(drift).upper() != "D1":
        return None
    if name not in GUARD_TOOLS:
        return None
    if not isinstance(result, dict) or "error" not in result:
        return None
    if "schema_violation" not in str(result.get("error", "")):
        return None
    if not isinstance(arguments, dict):
        return None

    if GUARD_FIXED_ARG in arguments:
        # A model that emits both names: drop the stale one and retry.
        if GUARD_STALE_ARG in arguments:
            return {
                key: value
                for key, value in arguments.items()
                if key != GUARD_STALE_ARG
            }
        return None

    if GUARD_STALE_ARG not in arguments:
        return None

    repaired = dict(arguments)
    repaired[GUARD_FIXED_ARG] = repaired.pop(GUARD_STALE_ARG)
    return repaired


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend(Protocol):
    name: str

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        ...


class TransformersBackend:
    """Base model in 4-bit plus an optional LoRA adapter. Greedy by default.

    Generation is stopped as soon as a complete `</tool_call>` block appears, and
    at the turn-end token if the model emits one first. Without that, `generate`
    happily keeps writing *past* the end of the model's turn — inventing the tool
    result and the next turn — up to `max_new_tokens` every single step. That is
    where the time went: a 20-task run at ~16 steps each was spending most of its
    tokens on hallucinated turns it never used.
    """

    def __init__(
        self,
        adapter: str | None = None,
        model_id: str = "CohereLabs/tiny-aya-global",
        load_in_4bit: bool = True,
        temperature: float = 0.0,
        max_context: int = 2048,
    ):
        import time

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.time = time
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.chat_template = TOOL_TEMPLATE
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {"device_map": {"": 0}}
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, **kwargs
            )
        except TypeError:  # transformers < 5 spells it torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, **kwargs
            )

        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

        self.model_id = model_id
        self.adapter = adapter
        self.temperature = temperature
        self.max_context = max_context
        self.name = f"transformers:{Path(adapter).name if adapter else 'base'}"

        self.stop_ids = self._stop_token_ids()
        self.generated_tokens = 0
        self.seconds = 0.0
        self.calls = 0

    def _stop_token_ids(self) -> list[int]:
        """EOS plus the turn-end token, so a finished turn ends the call."""
        ids = []
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        for token in ("<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != unk:
                ids.append(token_id)
        return sorted(set(ids))

    def _stopping_criteria(self, prompt_length: int):
        from transformers import StoppingCriteria, StoppingCriteriaList

        backend = self

        class StopOnToolCallEnd(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
                tail = input_ids[0][-12:]
                return "</tool_call>" in backend.tokenizer.decode(
                    tail, skip_special_tokens=True
                )

        return StoppingCriteriaList([StopOnToolCallEnd()])

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context,
        ).to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]

        sampling = self.temperature > 0
        started = self.time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=sampling,
                temperature=self.temperature if sampling else None,
                top_p=0.95 if sampling else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.stop_ids or None,
                stopping_criteria=self._stopping_criteria(prompt_length),
            )
        self.seconds += self.time.perf_counter() - started
        self.calls += 1
        self.generated_tokens += max(0, output[0].shape[0] - prompt_length)

        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # The template closes the turn with a special token; cut there so the
        # parser never sees the next turn's scaffold.
        for stopper in ("<|END_OF_TURN_TOKEN|>", "<|START_OF_TURN_TOKEN|>"):
            text = text.split(stopper)[0]
        return text.strip()


class ScriptedBackend:
    """Deterministic canned responses. Used by the tests, no torch needed."""

    name = "scripted"

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0
        self.seen_messages: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        self.seen_messages.append([dict(message) for message in messages])
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(bucket: str, allow_sealed: bool = False) -> list[dict[str, Any]]:
    if bucket == "final_sealed":
        if not allow_sealed:
            raise SystemExit(
                "Refusing to read the sealed set without --sealed-ok.\n"
                "The plan opens it once, on 12 Nov. Use --bucket drift_targeted "
                "or --bucket gate_regression until then."
            )
        from data.splits.sealed import load_sealed_tasks

        return [
            {"seed": task["seed"], "drift": task["drift"]}
            for task in load_sealed_tasks()
        ]

    path = ROOT / BUCKET_FILES[bucket]
    if not path.is_file():
        raise SystemExit(f"missing bucket file: {path}")
    with path.open(encoding="utf-8") as file:
        return [
            {"seed": json.loads(line)["seed"], "drift": json.loads(line).get("drift")}
            for line in file
            if line.strip()
        ]


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------

def run_episode(
    backend: Backend,
    seed: int,
    drift: str | None,
    *,
    prompt_suffix: str = "",
    prompt_variant: str = "clean",
    max_steps: int = MAX_STEPS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    drift_override: str | None = None,
    stale_schema: bool = False,
    repair_guard: bool = False,
) -> dict[str, Any]:
    """Generate one full trajectory against the real environment.

    `drift_override` forces the API version the model faces, whatever the task
    file says. That is what makes "how much does each drift cost this model on
    the same clean tasks" a single measurment instead of four bespoke files.
    """

    if drift_override is not None:
        drift = None if str(drift_override).lower() in {"clean", "none"} else str(drift_override).upper()

    snapshot = generate_snapshot(seed)
    truth = check_vulnerabilities(snapshot)
    env, schemas = build_runtime(CloudEnv(snapshot), drift=drift)

    # The registry the agent carries is a snapshot; the API moved underneath it.
    # With --stale-schema the prompt keeps the *clean* tool list while the
    # environment enforces v2 - which is how a real agent actually breaks, and
    # the only way an instruction-following model can be made to fail on a
    # rename it could otherwise read off the tool description.
    prompt_schemas = TOOL_SCHEMAS if stale_schema else schemas
    system = system_prompt_for(prompt_schemas)
    if prompt_suffix:
        # Any non-canonical system prompt is not the expert's, so the healer must
        # refuse to build a training pair from it. Derive the flag here rather
        # than trusting the caller to pass both arguments consistently.
        system = system + "\n\n" + prompt_suffix
        if prompt_variant == "clean":
            prompt_variant = "custom"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Find all vulnerable cloud resources."},
    ]
    calls: list[dict[str, Any]] = []
    invalid_calls = 0
    finished = False
    guard_repairs = 0
    guard_attempts = 0

    for _step in range(max_steps):
        raw = backend.generate(messages, max_new_tokens)
        messages.append({"role": "assistant", "content": raw})

        blocks = CALL_RE.findall(raw)
        if not blocks:
            invalid_calls += 1
            # One nudge per episode: tell the model the format, then stop if it
            # still will not call a tool. This is counted, never hidden.
            if invalid_calls == 1:
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "error": (
                                    "schema_violation: no <tool_call> block found. "
                                    'Reply with exactly <tool_call>{"name": ..., '
                                    '"arguments": {...}}</tool_call>'
                                )
                            }
                        ),
                    }
                )
                continue
            break

        block = blocks[0]
        try:
            parsed = json.loads(block)
            name = parsed.get("name")
            arguments = parsed.get("arguments", {})
        except json.JSONDecodeError:
            invalid_calls += 1
            break

        if not isinstance(name, str) or not isinstance(arguments, dict):
            invalid_calls += 1
            break

        calls.append({"name": name, "arguments": arguments})

        if name == "finish":
            finished = True
            answer = arguments.get("answer")
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "status": "success",
                            "answer": answer if isinstance(answer, list) else [],
                        }
                    ),
                }
            )
            break

        function = getattr(env, name, None)
        if callable(function):
            try:
                result = function(**arguments)
            except TypeError as exc:
                result = {"error": f"schema_violation: {exc}"}
        else:
            result = {"error": f"unknown_tool: Tool {name!r} not found"}

        if repair_guard and callable(function):
            repaired = _guard_repair(name, arguments, result, drift)
            if repaired is not None:
                guard_attempts += 1
                try:
                    retry = function(**repaired)
                except TypeError as exc:
                    retry = {"error": f"schema_violation: {exc}"}
                if isinstance(retry, dict) and "error" not in retry:
                    guard_repairs += 1
                    # The model sees the successful observation as if it had
                    # asked correctly. Keep the scoring trace consistent with
                    # that observation; record the repair for the article.
                    messages[-1] = {
                        "role": "assistant",
                        "content": _render_tool_call(name, repaired),
                    }
                    calls[-1] = {"name": name, "arguments": repaired}
                    result = retry
                else:
                    result = retry

        messages.append({"role": "tool", "content": json.dumps(result, default=str)})

    final_answer = []
    for call in reversed(calls):
        if call["name"] == "finish":
            value = call["arguments"].get("answer")
            final_answer = [str(item) for item in value] if isinstance(value, list) else []
            break

    verified = replay(snapshot, calls, drift)
    steps = len([call for call in calls if call["name"] != "finish"])

    redundant = 0
    seen: set[str] = set()
    for call in calls:
        if call["name"] == "finish":
            continue
        signature = json.dumps(call, sort_keys=True)
        if signature in seen:
            redundant += 1
        seen.add(signature)

    success = sorted(final_answer) == sorted(truth)
    gap = missing_evidence(snapshot, calls)

    return {
        "grounded_success": bool(success) and not gap,
        "missing_evidence": len(gap),
        "id": f"{drift or 'clean'}_{seed}",
        "task_id": f"{drift or 'clean'}_{seed}",
        "seed": seed,
        "split": "rollout",
        "drift": drift,
        "drift_override": drift_override,
        "stale_schema": stale_schema,
        "task": "Find all vulnerable cloud resources.",
        "template_id": "custom_tool_call_v1",
        "prompt_variant": prompt_variant,
        "messages": messages,
        "tool_calls": calls,
        "final_answer": final_answer,
        "family_success": family_success(final_answer, truth),
        "reward": verified["reward"],
        "success": success,
        "is_correct": success,
        "num_steps": steps,
        "valid_tool_calls": max(0, len(calls) - invalid_calls),
        "total_tool_calls": len(calls),
        "redundant_calls": redundant,
        "finished": finished,
        "error": None,
        "is_none_task": not truth,
        "invalid_calls": invalid_calls,
        "repair_guard": repair_guard,
        "guard_repairs": guard_repairs,
        "guard_attempts": guard_attempts,
        "ground_truth": truth,
    }


def to_record(trajectory: dict[str, Any], *, bucket: str, tag: str, variant: str) -> dict[str, Any]:
    """Project a trajectory onto the gate's record contract."""
    classification = classify(trajectory)
    return {
        "id": trajectory["id"],
        "task_id": trajectory["task_id"],
        "bucket": bucket,
        "seed": trajectory["seed"],
        "drift": trajectory["drift"] or "clean",
        "policy": tag,
        "prompt_variant": variant,
        "stale_schema": trajectory.get("stale_schema", False),
        "success": trajectory["success"],
        # Correct *and* every required evidence call made. The gate compares
        # efficiency on these, because an ungrounded success is right by luck.
        "grounded_success": trajectory.get("grounded_success", False),
        "missing_evidence": trajectory.get("missing_evidence", 0),
        "num_steps": trajectory["num_steps"],
        "total_tool_calls": trajectory["total_tool_calls"],
        "redundant_calls": trajectory["redundant_calls"],
        "family_success": trajectory["family_success"],
        "failure_mode": classification["primary"],
        "reward": trajectory["reward"],
        "invalid_calls": trajectory["invalid_calls"],
        "is_none_task": trajectory["is_none_task"],
        "drift_inferred": classification["drift_inferred"],
        "repair_guard": trajectory.get("repair_guard", False),
        "guard_repairs": trajectory.get("guard_repairs", 0),
        "guard_attempts": trajectory.get("guard_attempts", 0),
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    none = [row for row in records if row["is_none_task"]]
    vulnerable = [row for row in records if not row["is_none_task"]]
    successes = [row for row in records if row["success"]]
    steps = [row["num_steps"] for row in successes]

    def rate(rows: list[dict[str, Any]]) -> float | None:
        return sum(1 for row in rows if row["success"]) / len(rows) if rows else None

    none_rate = rate(none)
    vuln_rate = rate(vulnerable)
    balanced = (
        (none_rate + vuln_rate) / 2
        if none_rate is not None and vuln_rate is not None else None
    )

    return {
        "n": len(records),
        "task_success": len(successes) / len(records) if records else 0.0,
        "balanced_success": balanced,
        "success_on_none": none_rate,
        "success_on_vulnerable": vuln_rate,
        "median_steps": statistics.median(steps) if steps else None,
        "call_rate": (
            sum(1 for row in records if row["total_tool_calls"] > 0) / len(records)
            if records else 0.0
        ),
        "invalid_call_tasks": sum(1 for row in records if row["invalid_calls"] > 0),
        "guard_repairs": sum(row.get("guard_repairs", 0) for row in records),
        "guard_tasks": sum(1 for row in records if row.get("guard_repairs", 0) > 0),
        "labels": {
            label: sum(1 for row in records if row["failure_mode"] == label)
            for label in sorted({row["failure_mode"] for row in records if row["failure_mode"]})
        },
    }


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n  {title}")
    print(
        f"    n={summary['n']}  task_success={summary['task_success']:.3f}  "
        f"balanced={summary['balanced_success']:.3f}  "
        f"(none {summary['success_on_none']:.3f} / "
        f"vulnerable {summary['success_on_vulnerable']:.3f})"
    )
    steps = summary["median_steps"]
    print(
        f"    median_steps={steps if steps is not None else 'n/a'}  "
        f"tasks_that_called_a_tool={summary['call_rate']:.1%}  "
        f"tasks_with_a_malformed_call={summary['invalid_call_tasks']}"
    )
    if summary.get("guard_repairs"):
        print(
            f"    guard repairs={summary['guard_repairs']} "
            f"across {summary['guard_tasks']} task(s)"
        )
    if summary["labels"]:
        labels = ", ".join(f"{name}x{count}" for name, count in summary["labels"].items())
        print(f"    failure labels: {labels}")
        if "residual" in summary["labels"]:
            share = summary["labels"]["residual"] / summary["n"]
            print(f"    residual share: {share:.3f} (counted, never sent to an LLM)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _display(path: Path) -> str:
    """Relative to the repo when possible, absolute otherwise.

    The old code called `path.relative_to(ROOT)` unconditionally, which raises
    ValueError when `--trajectories-dir` is a relative path or lives outside the
    checkout. It crashed after the files were written, which made a successful
    run look like a failure.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _output_paths(
    bucket: str,
    tag: str,
    variant: str,
    trajectories_dir: Path | None = None,
    records_dir: Path | None = None,
) -> tuple[Path, Path]:
    suffix = "" if variant == "clean" else f"_{variant}"
    records = (records_dir or RUNS) / f"{tag}{suffix}_{bucket}.jsonl"
    directory = trajectories_dir or TRAJECTORIES
    if not directory.is_absolute():
        directory = ROOT / directory
    trajectories = directory / f"{tag}{suffix}_{bucket}.jsonl"
    return records, trajectories


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _run_variant(
    backend: Backend,
    tasks: list[dict[str, Any]],
    bucket: str,
    tag: str,
    variant: str,
    prompt_suffix: str,
    limit: int | None,
    max_steps: int,
    max_new_tokens: int,
    trajectories_dir: Path | None = None,
    drift_override: str | None = None,
    filename_label: str | None = None,
    records_dir: Path | None = None,
    stale_schema: bool = False,
    repair_guard: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = tasks[:limit] if limit else tasks
    trajectories = []
    records = []

    for position, task in enumerate(selected, start=1):
        trajectory = run_episode(
            backend,
            task["seed"],
            task["drift"],
            prompt_suffix=prompt_suffix,
            prompt_variant=variant,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            drift_override=drift_override,
            stale_schema=stale_schema,
            repair_guard=repair_guard,
        )
        trajectories.append(trajectory)
        records.append(
            to_record(
                trajectory, bucket=bucket, tag=tag, variant=variant
            )
        )
        if position % 5 == 0 or position == len(selected):
            wins = sum(1 for row in records if row["success"])
            print(
                f"    {filename_label or variant}: {position}/{len(selected)} tasks, "
                f"success {wins}/{position}",
                flush=True,
            )

    records_path, trajectories_path = _output_paths(
        bucket, tag, filename_label or variant, trajectories_dir, records_dir
    )
    _write_jsonl(records_path, records)
    _write_jsonl(trajectories_path, trajectories)
    print(f"    records     -> {_display(records_path)}")
    print(f"    trajectories-> {_display(trajectories_path)}")
    return records, trajectories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a model against a gate bucket and emit gate-ready records."
    )
    parser.add_argument("--adapter", help="LoRA adapter directory. Omit for the base model.")
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--bucket", default="drift_targeted", choices=list(BUCKET_FILES))
    parser.add_argument("--tag", default="model", help="Name used in the output files.")
    parser.add_argument(
        "--prompt-variant",
        default="clean",
        choices=list(PROMPT_VARIANTS),
        help="Which one-sentence prompt arm to use.",
    )
    parser.add_argument("--prompt-suffix", default="", help="Free-form extra system text.")
    parser.add_argument(
        "--prompt-only-control",
        action="store_true",
        help="Run clean and prompt_only_d1 on the same seeds and print the delta.",
    )
    parser.add_argument("--limit", type=int, help="Only the first N tasks (smoke run).")
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        help="Where to write full trajectories. Use data/drift_train so "
             "`heal.examples --source rollouts` picks them up in place.",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-4bit", action="store_true", help="Load in fp16/bf16.")
    parser.add_argument(
        "--compare-to",
        help="Baseline records prefix, e.g. artifacts/runs/v0. Runs the gate on both buckets.",
    )
    parser.add_argument(
        "--drift-override",
        choices=["clean", "D1", "D2", "D3", "D4", "all"],
        help="Force the API version the model faces. 'all' sweeps clean + D1-D4 "
             "over the same tasks and prints the damage table (Stage 1.6).",
    )
    parser.add_argument(
        "--stale-schema",
        action="store_true",
        help="Keep the CLEAN tool description in the system prompt while the "
             "environment enforces the drifted API. Models the real failure: a "
             "pinned tool registry against a moved API.",
    )
    parser.add_argument(
        "--repair-guard",
        action="store_true",
        help="D1 runtime guard: when a bucket tool rejects `bucket`, retry once "
             "with `bucket_name` and show the model the successful result. "
             "Clean traffic is never touched. Repairs are recorded per task.",
    )
    parser.add_argument(
        "--bucket-label",
        choices=["drift_targeted", "gate_regression", "dev"],
        help="Label a --seeds run as one of the gate buckets, so a fresh, never-"
             "inspected block can be scored by the gate without reusing the "
             "bucket's own seeds.",
    )
    parser.add_argument(
        "--seeds",
        help="START-END seed range to use instead of a bucket file, e.g. "
             "30000-30059. Use with --drift-override to make a drift-train set "
             "that is disjoint from every evaluation bucket.",
    )
    parser.add_argument(
        "--sealed-ok",
        action="store_true",
        help="Permit reading the sealed set. The plan opens it once, on 12 Nov.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and stop.")
    args = parser.parse_args()

    if args.seeds:
        try:
            start_text, end_text = args.seeds.split("-")
            start, end = int(start_text), int(end_text)
        except ValueError as exc:
            raise SystemExit(f"--seeds must be START-END, got {args.seeds!r}") from exc
        tasks = [{"seed": seed, "drift": None} for seed in range(start, end + 1)]
        args.bucket = args.bucket_label or "drift_train"
        print(
            f"seeds {start}-{end} ({len(tasks)} tasks) -> bucket label "
            f"'{args.bucket}'"
        )
        if args.drift_override in (None, "all"):
            raise SystemExit(
                "--seeds needs an explicit --drift-override D1..D4 (or clean); "
                "otherwise the task list has no API version."
            )
    else:
        tasks = load_tasks(args.bucket, allow_sealed=args.sealed_ok)
    if args.limit:
        tasks = tasks[: args.limit]

    # A stale description matters per task: on a clean task there is no drift to
    # be stale about, so gate_regression (42 clean + D2/D3/D4) needs no override
    # at all. Forcing one there would relabel every clean task as drifted and
    # silently destroy the regression bucket, which is the opposite of safe.
    if args.stale_schema and args.drift_override == "clean":
        raise SystemExit(
            "--stale-schema with --drift-override clean is a contradiction: a "
            "stale description only matters when the API underneath it moved."
        )
    if args.stale_schema and args.seeds and not args.drift_override:
        raise SystemExit(
            "--stale-schema with --seeds needs --drift-override D1..D4: a bare "
            "seed range has no API version of its own."
        )


    print(
        f"bucket={args.bucket}  tasks={len(tasks)}  "
        f"adapter={args.adapter or 'base'}  decoding="
        f"{'greedy' if args.temperature == 0 else f'temperature {args.temperature}'}"
    )
    print(f"prompt variant(s): ", end="")
    variants = (
        ["clean", "prompt_only_d1"] if args.prompt_only_control
        else [args.prompt_variant]
    )
    print(", ".join(variants))
    if args.dry_run:
        print("dry run: stopping before the model loads.")
        return

    backend = TransformersBackend(
        adapter=args.adapter,
        model_id=args.model,
        load_in_4bit=not args.no_4bit,
        temperature=args.temperature,
    )
    print(f"backend: {backend.name}")

    results: dict[str, list[dict[str, Any]]] = {}
    written: dict[str, Path] = {}

    if args.drift_override == "all":
        if args.bucket != "dev":
            print(
                f"warning: sweeping drift over `--bucket {args.bucket}`. The "
                "clean arm is only a fair baseline when the tasks themselves "
                "are clean; `--bucket dev` is the intended use."
            )
        # One sweep, one table: the same tasks under clean and under each drift.
        # This is Stage 1.6 of the plan — "each drift must cost at least 20
        # points" — measured on the model instead of asserted.
        damage: dict[str, dict[str, Any]] = {}
        for forced in ("clean", "D1", "D2", "D3", "D4"):
            records, _trajectories = _run_variant(
                backend,
                tasks,
                args.bucket,
                args.tag,
                variants[0],
                PROMPT_VARIANTS[variants[0]] if args.prompt_suffix == "" else args.prompt_suffix,
                None,
                args.max_steps,
                args.max_new_tokens,
                args.trajectories_dir,
                drift_override=forced,
                filename_label=f"{variants[0]}_{forced}"
                + ("_stale" if args.stale_schema else ""),
                stale_schema=args.stale_schema,
                repair_guard=args.repair_guard,
            )
            damage[forced] = summarise(records)

        clean_success = damage["clean"]["task_success"]
        print("\n  drift damage (same tasks, model forced onto each API version)")
        print("    | condition | task_success | balanced | drop vs clean | 20pt bar |")
        for condition, summary in damage.items():
            drop = clean_success - summary["task_success"]
            bar = "n/a" if condition == "clean" else ("yes" if drop >= 0.20 else "NO")
            print(
                f"    | {condition:9s} | {summary['task_success']:.3f}        | "
                f"{summary['balanced_success']:.3f}    | {drop:+.3f}        | {bar} |"
            )
        (RUNS / f"{args.tag}_drift_damage.json").write_text(
            json.dumps(damage, indent=2), encoding="utf-8"
        )
        print(f"\n  wrote {RUNS / f'{args.tag}_drift_damage.json'}")
        return

    forced = args.drift_override
    for variant in variants:
        suffix = PROMPT_VARIANTS[variant] if args.prompt_suffix == "" else args.prompt_suffix
        label = variant if not forced else f"{variant}_{forced}"
        if args.stale_schema:
            label = f"{label}_stale"
        records, _trajectories = _run_variant(
            backend,
            tasks,
            args.bucket,
            args.tag,
            variant,
            suffix,
            None,
            args.max_steps,
            args.max_new_tokens,
            args.trajectories_dir,
            drift_override=forced,
            filename_label=label,
            stale_schema=args.stale_schema,
            repair_guard=args.repair_guard,
        )
        results[variant] = records
        written[variant] = _output_paths(args.bucket, args.tag, label)[0]
        print_summary(f"summary: {variant}{' @ ' + forced if forced else ''}", summarise(records))

    if getattr(backend, "calls", 0):
        per_call = backend.seconds / backend.calls
        rate = backend.generated_tokens / backend.seconds if backend.seconds else 0.0
        print(
            f"\n  timing: {backend.calls} generations, "
            f"{backend.seconds:.0f}s total, {per_call:.2f}s/call, "
            f"{backend.generated_tokens} tokens, {rate:.1f} tok/s"
        )
        print(
            f"          {backend.generated_tokens / backend.calls:.0f} tokens/call "
            "(a complete tool call is ~40-60; much more means the model is "
            "writing past its turn)"
        )

    if args.prompt_only_control and len(results) == 2:
        clean, prompted = results["clean"], results["prompt_only_d1"]
        clean_success = sum(1 for row in clean if row["success"]) / len(clean)
        prompted_success = sum(1 for row in prompted if row["success"]) / len(prompted)
        print("\n  prompt-only control (same adapter, same seeds)")
        print(f"    clean prompt      : {clean_success:.3f}")
        print(f"    prompt + D1 hint  : {prompted_success:.3f}")
        print(f"    delta             : {prompted_success - clean_success:+.3f}")
        print(
            "    Write this down whichever way it lands. A prompt that fixes D1\n"
            "    is a finding, not an embarrassment; a prompt that does not is the\n"
            "    argument for the retrain."
        )

    if args.compare_to:
        from gates.deployment_gate import evaluate_deployment_gate
        from gates.registry import log_decision

        # Gate the candidate against the baseline. With --prompt-only-control the
        # clean arm is the candidate, so the prompt-only arm can be read as the
        # control rather than silently replacing the model's real score.
        gate_variant = "clean" if "clean" in written else variants[-1]
        candidate_prefix = written[gate_variant]

        baseline: list[dict[str, Any]] = []
        candidate: list[dict[str, Any]] = []
        for bucket in ("drift_targeted", "gate_regression"):
            base_path = ROOT / f"{args.compare_to}_{bucket}.jsonl"
            cand_path = RUNS / f"{args.tag}_{bucket}.jsonl"
            if gate_variant != "clean":
                cand_path = _output_paths(bucket, args.tag, gate_variant)[0]
            if not base_path.is_file() or not cand_path.is_file():
                print(f"\n  gate: skipping, missing {base_path.name} or {cand_path.name}")
                return
            with base_path.open(encoding="utf-8") as file:
                baseline += [json.loads(line) for line in file if line.strip()]
            with cand_path.open(encoding="utf-8") as file:
                candidate += [json.loads(line) for line in file if line.strip()]

        ages = {row.get("stale_schema", False) for row in baseline}
        ages |= {row.get("stale_schema", False) for row in candidate}
        if len(ages) > 1:
            print(
                "\n  gate: refusing to compare a stale-schema run with a "
                "fresh-schema run. The baseline and the candidate must see the "
                "same tool description, or the gain measures the prompt rather "
                "than the retrain."
            )
            return

        decision = evaluate_deployment_gate(baseline, candidate)
        log_decision(
            decision,
            baseline=args.compare_to,
            candidate=f"{args.tag} ({gate_variant})",
            meta={
                "adapter": args.adapter,
                "prompt_variant": gate_variant,
                "stale_schema": sorted(ages)[0],
                "candidate_records": str(candidate_prefix.relative_to(ROOT)),
            },
        )
        print(f"\n  gate: {decision['decision']}")
        print(f"  reason: {decision['reason']}")


if __name__ == "__main__":
    main()
