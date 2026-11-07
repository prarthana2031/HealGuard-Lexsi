"""Leakage, reproducibility and answer-balance tests. World v2.

What changed from the v1 file:

* every split is now selected to be exactly half `none`, so the balance test
  asserts the construction instead of hoping for it;
* the seed ledger is checked, including the reserved blocks (classifier battery,
  healer simulator, archived v1 rollouts);
* the sealed set is regenerated in the current contract, so the "frozen file is
  a task list" exemption is gone;
* the archived world-v1 rollouts are checked for collision but never mixed in.
"""

import hashlib
import json
import unittest
from pathlib import Path

from oracle.checker import check_vulnerabilities
from world.snapshot import generate_snapshot

ROOT = Path(__file__).resolve().parents[1]

SPLITS = {
    "train_clean": "data/train_clean/expert_traces.jsonl",
    "dev": "data/dev/expert_traces.jsonl",
    "drift_targeted": "data/drift_targeted/expert_traces.jsonl",
    "gate_regression": "data/gate_regression/expert_traces.jsonl",
    "final_sealed": "data/final_sealed/sealed_traces.jsonl",
}

SCORER_FIELDS = {
    "id",
    "seed",
    "split",
    "task",
    "messages",
    "tool_calls",
    "final_answer",
    "success",
    "num_steps",
    "valid_tool_calls",
    "total_tool_calls",
    "redundant_calls",
    "template_id",
}


def rows(relative: str) -> list[dict]:
    path = ROOT / relative
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class TestLeakage(unittest.TestCase):

    def test_no_seed_appears_in_two_splits(self):
        seen: dict[int, str] = {}
        for name, relative in SPLITS.items():
            for row in rows(relative):
                seed = row["seed"]
                self.assertNotIn(
                    seed, seen,
                    f"seed {seed} is in both {seen.get(seed)} and {name}",
                )
                seen[seed] = name

    def test_ledger_matches_the_files_on_disk(self):
        ledger = json.loads(
            (ROOT / "data" / "splits" / "seed_ledger.json").read_text(encoding="utf-8")
        )
        for name, relative in SPLITS.items():
            on_disk = [row["seed"] for row in rows(relative)]
            self.assertEqual(
                sorted(on_disk),
                sorted(ledger["seeds"][name]),
                f"{name} does not match the ledger",
            )

    def test_reserved_blocks_do_not_touch_any_split(self):
        ledger = json.loads(
            (ROOT / "data" / "splits" / "seed_ledger.json").read_text(encoding="utf-8")
        )
        split_seeds = {seed for seeds in ledger["seeds"].values() for seed in seeds}
        for name, (start, size) in ledger["reserved"].items():
            overlap = split_seeds & set(range(start, start + size))
            self.assertFalse(overlap, f"{name} overlaps splits at {sorted(overlap)[:5]}")

    def test_archived_world_v1_rollouts_do_not_collide(self):
        archive = ROOT / "data" / "archive" / "world_v1" / "v0_failed_rollouts.jsonl"
        if not archive.is_file():
            self.skipTest("no archive")
        split_seeds = {row["seed"] for name in SPLITS for row in rows(SPLITS[name])}
        with archive.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                self.assertNotIn(json.loads(line)["seed"], split_seeds)

    def test_sealed_hash_matches_the_freeze_record(self):
        sealed = ROOT / "data" / "final_sealed" / "sealed_traces.jsonl"
        recorded = (ROOT / "data" / "splits" / "sealed_hash.txt").read_text().strip()
        content = sealed.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(content).hexdigest(), recorded)

        freeze = (ROOT / "data" / "splits" / "sealed_freeze.md").read_text(encoding="utf-8")
        self.assertIn(recorded, freeze)
        self.assertIn("Opened before any re-freeze?** No", freeze)

        # Append-only: the world-v1 hash must still be in the history.
        history = json.loads(
            (ROOT / "data" / "splits" / "sealed_history.json").read_text(encoding="utf-8")
        )
        self.assertEqual(history[0]["version"], "world v1")
        self.assertEqual(history[-1]["sha256"], recorded)
        self.assertNotEqual(history[0]["sha256"], history[-1]["sha256"])

    def test_sealed_set_was_never_used_to_build_examples(self):
        heal = ROOT / "data" / "heal" / "manifest.json"
        if not heal.is_file():
            self.skipTest("no heal set yet")
        sealed = {row["seed"] for row in rows(SPLITS["final_sealed"])}
        for pair_rows in (ROOT / "data" / "heal").glob("*.jsonl"):
            with pair_rows.open(encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    seed = row.get("provenance", {}).get("seed") or row.get("seed")
                    if seed is not None:
                        self.assertNotIn(seed, sealed)


class TestWorldV2Shape(unittest.TestCase):
    """The properties that motivated the redesign, asserted not assumed."""

    def test_splits_are_exactly_half_none(self):
        for name, relative in SPLITS.items():
            split = rows(relative)
            none = sum(1 for row in split if not row["final_answer"])
            with self.subTest(split=name):
                self.assertEqual(none * 2, len(split), f"{name}: {none}/{len(split)}")

    def test_answer_sets_vary_in_size(self):
        sizes = set()
        for relative in SPLITS.values():
            for row in rows(relative):
                sizes.add(len(row["final_answer"]))
        # v1 could only ever produce 0 or 3.
        self.assertGreaterEqual(len(sizes), 4, f"only sizes {sorted(sizes)}")
        self.assertIn(1, sizes, "no single-finding task exists")

    def test_vulnerability_is_not_positional(self):
        counts: dict[int, int] = {}
        for relative in SPLITS.values():
            for row in rows(relative):
                for resource in row["final_answer"]:
                    index = int(resource.split("-")[-1])
                    counts[index] = counts.get(index, 0) + 1
        # Index 0 must not dominate: v1 always put the finding at index 0.
        total = sum(counts.values())
        self.assertLess(counts.get(0, 0) / total, 0.45, counts)

    def test_families_are_decoupled(self):
        """At least one split must contain a single-family task."""
        single = 0
        for relative in SPLITS.values():
            for row in rows(relative):
                families = {
                    "F1" if r.startswith("bucket-")
                    else "F2" if r.startswith("role-")
                    else "F4"
                    for r in row["final_answer"]
                }
                if len(families) == 1:
                    single += 1
        self.assertGreater(single, 0)

    def test_every_split_exposes_the_scorer_contract(self):
        for name, relative in SPLITS.items():
            for row in rows(relative)[:5]:
                with self.subTest(split=name, task=row.get("id")):
                    missing = SCORER_FIELDS - set(row)
                    self.assertFalse(missing, f"{name}: missing {missing}")

    def test_sealed_set_is_a_usable_task_list(self):
        from data.splits.sealed import load_sealed_tasks, verify_sealed_hash

        self.assertTrue(verify_sealed_hash())
        tasks = load_sealed_tasks()
        self.assertEqual(len(tasks), 50)
        for task in tasks:
            self.assertIsInstance(task["truth"], list)
            self.assertIsInstance(task["seed"], int)


class TestDeterminism(unittest.TestCase):

    def test_same_seed_gives_byte_identical_snapshots(self):
        first = json.dumps(generate_snapshot(42), sort_keys=True)
        second = json.dumps(generate_snapshot(42), sort_keys=True)
        self.assertEqual(first, second)

    def test_same_seed_gives_the_same_ground_truth(self):
        self.assertEqual(
            check_vulnerabilities(generate_snapshot(7)),
            check_vulnerabilities(generate_snapshot(7)),
        )


if __name__ == "__main__":
    unittest.main()
