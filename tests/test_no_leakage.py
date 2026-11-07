"""Leakage, reproducibility and answer-balance tests.

The old file checked only that the sealed seeds did not appear in four splits.
It missed the leak that actually mattered: the *real model failures* in
`drift_train` were on seeds 2000-2019, the same seeds `drift_targeted` used, so
the healer trained on the exact tasks the gate grades.

It also assumed the traces carried a `template_id`; they did not, so template
disjointness was never checked at all.
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

    def test_real_model_failures_do_not_touch_the_gate_buckets(self):
        """The healer must not train on the tasks the gate grades."""
        bucket_seeds = {
            row["seed"]
            for relative in (
                "data/drift_targeted/expert_traces.jsonl",
                "data/gate_regression/expert_traces.jsonl",
            )
            for row in rows(relative)
        }

        failures = ROOT / "data" / "drift_train" / "v0_failed_rollouts.jsonl"
        if not failures.is_file():
            self.skipTest("no real rollouts")

        with failures.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                seed = json.loads(line)["seed"]
                self.assertNotIn(seed, bucket_seeds)

    def test_sealed_hash_matches_the_committed_hash(self):
        sealed = ROOT / "data" / "final_sealed" / "sealed_traces.jsonl"
        recorded = (ROOT / "data" / "splits" / "sealed_hash.txt").read_text().strip()
        content = sealed.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(content).hexdigest(), recorded)

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
                    seed = (
                        row.get("provenance", {}).get("seed")
                        or row.get("seed")
                    )
                    if seed is not None:
                        self.assertNotIn(seed, sealed)


class TestAnswerBalance(unittest.TestCase):

    def test_none_rate_is_between_40_and_60_percent_in_every_split(self):
        for name, relative in SPLITS.items():
            split = rows(relative)
            none = sum(1 for row in split if not row["final_answer"])
            share = none / len(split)
            with self.subTest(split=name, share=share):
                self.assertGreaterEqual(share, 0.40)
                self.assertLessEqual(share, 0.60)

    def test_every_split_exposes_the_scorer_contract(self):
        """The frozen sealed file is exempt: it is a task list, not a trace set."""
        for name, relative in SPLITS.items():
            if name == "final_sealed":
                continue
            for row in rows(relative)[:5]:
                with self.subTest(split=name, task=row.get("id")):
                    missing = SCORER_FIELDS - set(row)
                    self.assertFalse(missing, f"{name}: missing {missing}")

    def test_sealed_set_is_a_usable_task_list(self):
        from data.splits.sealed import load_sealed_tasks, verify_sealed_hash

        self.assertTrue(verify_sealed_hash())
        tasks = load_sealed_tasks()
        self.assertTrue(tasks)
        for task in tasks:
            self.assertIsInstance(task["truth"], list)
            self.assertIsInstance(task["seed"], int)

    def test_template_ids_are_recorded(self):
        """One template exists; the test records that rather than pretending."""
        templates = {row["template_id"] for row in rows(SPLITS["dev"])}
        self.assertTrue(templates)
        self.assertNotIn(None, templates)


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
