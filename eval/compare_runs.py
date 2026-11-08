"""Compare two runs on the same tasks, as a paired test.

Why this exists: at N=20 a difference of one or two tasks is noise, and picking
the checkpoint with the highest raw score is picking the noisiest estimate. The
right question is not "which score is bigger" but "on the tasks where they
disagree, which one wins more often".

That is McNemar's test, and it only looks at the discordant pairs, so it is much
more sensitive than comparing two proportions.

    python -m eval.compare_runs artifacts/runs/v0_ck150_dev.jsonl \
                               artifacts/runs/v0_ck250_dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def summarise(rows: dict[str, dict], ids: list[str]) -> dict:
    subset = [rows[i] for i in ids]
    none = [r for r in subset if r.get("is_none_task")]
    vuln = [r for r in subset if not r.get("is_none_task")]
    success = [r for r in subset if r["success"]]
    steps = [r["num_steps"] for r in success]

    def rate(group):
        return sum(1 for r in group if r["success"]) / len(group) if group else None

    none_rate = rate(none)
    vuln_rate = rate(vuln)
    balanced = (
        (none_rate + vuln_rate) / 2
        if none_rate is not None and vuln_rate is not None else None
    )
    return {
        "n": len(subset),
        "task_success": len(success) / len(subset) if subset else 0.0,
        "balanced": balanced,
        "success_on_none": none_rate,
        "success_on_vulnerable": vuln_rate,
        "median_steps": statistics.median(steps) if steps else None,
        "redundant_rate": (
            sum(r["redundant_calls"] for r in subset)
            / max(1, sum(r["total_tool_calls"] for r in subset))
        ),
    }


def sign_test(b: int, c: int) -> float:
    """Two-sided exact binomial p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(first: dict[str, dict], second: dict[str, dict]) -> dict:
    shared = sorted(set(first) & set(second))
    if not shared:
        raise SystemExit("the two runs share no task ids; were they run on the same bucket?")

    only_first = [i for i in shared if first[i]["success"] and not second[i]["success"]]
    only_second = [i for i in shared if second[i]["success"] and not first[i]["success"]]
    both = sum(1 for i in shared if first[i]["success"] and second[i]["success"])
    neither = sum(1 for i in shared if not first[i]["success"] and not second[i]["success"])

    return {
        "n": len(shared),
        "a": summarise(first, shared),
        "b": summarise(second, shared),
        "both": both,
        "neither": neither,
        "only_a": len(only_first),
        "only_b": len(only_second),
        "only_a_ids": only_first,
        "only_b_ids": only_second,
        "p_value": sign_test(len(only_first), len(only_second)),
    }


def canary_warning(result: dict, label_a: str, label_b: str) -> str | None:
    """A run that never solves a vulnerable task is the always-none canary.

    With a ~50% `none` rate, answering `[]` everywhere scores 0.500 task success.
    That is not a model; it is the canary, and it must never be read as clearing
    the 0.50 floor.
    """
    collapsed = []
    for label, summary in ((label_a, result["a"]), (label_b, result["b"])):
        if summary["success_on_vulnerable"] == 0.0:
            collapsed.append(label)
    if not collapsed:
        return None
    return (
        f"CANARY ALERT: {', '.join(collapsed)} solved 0 vulnerable tasks. "
        "Task success on this bucket equals the `always-none` canary "
        "(see artifacts/canaries.json), not a working agent. Judge it by "
        "success_on_vulnerable, not by task_success."
    )


def verdict(result: dict, label_a: str, label_b: str) -> str:
    p = result["p_value"]
    delta = result["b"]["task_success"] - result["a"]["task_success"]

    if result["a"]["success_on_vulnerable"] == 0.0 and result["b"]["success_on_vulnerable"] == 0.0:
        return (
            "both runs are collapsed to `always-none`; there is nothing to "
            "compare and no checkpoint to keep. Retrain, and check the training "
            "log for the `mask ok: ... supervised` line and the planned step count."
        )

    if p >= 0.05:
        return (
            f"no evidence they differ (p={p:.3f}). Treat as a tie and decide on "
            f"balanced success, then on median steps. Do not pick a winner from a "
            f"{abs(delta) * 100:.0f}-point gap at n={result['n']}."
        )
    winner = label_b if delta > 0 else label_a
    return f"{winner} is better on the discordant pairs (p={p:.3f}). Use {winner}."


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of two runs.")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    first, second = load(ROOT / args.a), load(ROOT / args.b)
    result = compare(first, second)

    print(f"shared tasks: {result['n']}")
    print(
        f"\n{'':<24}{args.label_a:>12}{args.label_b:>12}"
    )
    for field in (
        "task_success",
        "balanced",
        "success_on_none",
        "success_on_vulnerable",
        "median_steps",
        "redundant_rate",
    ):
        a, b = result["a"][field], result["b"][field]
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
        print(f"{field:<24}{fmt(a):>12}{fmt(b):>12}")

    print(
        f"\ndiscordant: {args.label_a} only {result['only_a']}, "
        f"{args.label_b} only {result['only_b']}; "
        f"both {result['both']}, neither {result['neither']}"
    )
    print(f"sign test (McNemar): p={result['p_value']:.4f}")

    warning = canary_warning(result, args.label_a, args.label_b)
    if warning:
        print(f"\n{warning}")

    print(f"\nverdict: {verdict(result, args.label_a, args.label_b)}")

    if result["only_a_ids"]:
        print(f"  {args.label_a} only: {', '.join(result['only_a_ids'][:8])}")
    if result["only_b_ids"]:
        print(f"  {args.label_b} only: {', '.join(result['only_b_ids'][:8])}")


if __name__ == "__main__":
    main()
