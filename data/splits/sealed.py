"""Read the sealed set as a task list, not as a trace bundle.

The sealed file is hash-locked and was generated before the trajectory contract
settled, so its rows carry `task_id`/`is_correct` rather than the scorer fields.
That is fine: the sealed set's job is to supply **tasks and their ground truth**.
The agent's own traces are what gets scored, and the oracle recomputes the truth
from the seed anyway.

Rewriting the file to make a schema test pass would break the committed sha256
and move the goalposts. Instead the file is verified and read through this
module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from oracle.checker import check_vulnerabilities
from world.snapshot import generate_snapshot

ROOT = Path(__file__).resolve().parents[2]
SEALED = ROOT / "data" / "final_sealed" / "sealed_traces.jsonl"
RECORDED_HASH = ROOT / "data" / "splits" / "sealed_hash.txt"


def sealed_hash() -> str:
    content = SEALED.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def verify_sealed_hash() -> bool:
    return sealed_hash() == RECORDED_HASH.read_text().strip()


def load_sealed_tasks(verify: bool = True) -> list[dict]:
    """Return `{task_id, seed, truth}` for every sealed task."""
    if verify and not verify_sealed_hash():
        raise RuntimeError(
            "sealed split hash does not match the committed hash — it has been "
            "modified. Do not score it."
        )

    tasks = []
    with SEALED.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            seed = row["seed"]
            truth = check_vulnerabilities(generate_snapshot(seed))
            stored = row.get("final_answer")
            if stored is not None and sorted(stored) != sorted(truth):
                raise RuntimeError(
                    f"sealed task {seed}: stored answer disagrees with the oracle"
                )
            tasks.append(
                {
                    "task_id": row.get("task_id") or row.get("id"),
                    "seed": seed,
                    "truth": truth,
                    "drift": row.get("drift"),
                }
            )
    return tasks
