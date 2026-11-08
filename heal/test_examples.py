"""The plan's central CI assertion.

"Assert in CI that 100% of emitted chosen traces replay to reward 1.  This
assertion is the project.  If it is annoying, it is working."

These tests do not trust the manifest.  They re-open the emitted files, rebuild
each full trajectory, execute every tool call against a fresh environment and
recompute the oracle.
"""

import json
import unittest
from pathlib import Path

from heal.replay import parse_calls, replay
from world.snapshot import generate_snapshot

ROOT = Path(__file__).resolve().parents[1]
HEAL = ROOT / "data" / "heal"


def load(name):
    path = HEAL / name
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class TestCorrectiveExamples(unittest.TestCase):

    def setUp(self):
        self.pairs = load("dpo_pairs.jsonl")
        if self.pairs is None:
            self.skipTest("run `python -m heal.examples --drift D1` first")

    def test_every_chosen_trace_replays_to_a_perfect_reward(self):
        for pair in self.pairs:
            seed = pair["provenance"]["seed"]
            drift = pair["provenance"]["drift"]
            messages = pair["prompt"] + pair["chosen"]
            result = replay(generate_snapshot(seed), parse_calls(messages), drift)
            with self.subTest(seed=seed):
                self.assertTrue(
                    result["correct"],
                    f"chosen answer {result['answer']} != truth {result['truth']}",
                )
                self.assertTrue(
                    result["perfect"], f"imperfect replay: {result['errors']}"
                )

    def test_every_rejected_side_is_actually_wrong(self):
        for pair in self.pairs:
            seed = pair["provenance"]["seed"]
            drift = pair["provenance"]["drift"]
            messages = pair["prompt"] + pair["rejected"]
            result = replay(generate_snapshot(seed), parse_calls(messages), drift)
            with self.subTest(seed=seed):
                self.assertFalse(result["correct"])

    def test_pair_differs_only_after_the_shared_prefix(self):
        for pair in self.pairs:
            self.assertIsInstance(pair["prompt"], list)
            self.assertTrue(pair["prompt"])
            self.assertNotEqual(pair["chosen"], pair["rejected"])
            # The first differing message must be an assistant call, not a
            # system/user/tool message: the prefix is genuinely shared.
            self.assertEqual(pair["chosen"][0]["role"], "assistant")
            self.assertEqual(pair["rejected"][0]["role"], "assistant")

    def test_provenance_is_complete(self):
        required = {
            "source_run",
            "label",
            "drift",
            "seed",
            "divergence_step",
            "template_id",
            "replay_hash",
        }
        for pair in self.pairs:
            with self.subTest(seed=pair["provenance"]["seed"]):
                self.assertTrue(
                    required.issubset(pair["provenance"]),
                    required - set(pair["provenance"]),
                )

    def test_clean_mix_is_between_30_and_50_percent(self):
        manifest = json.loads((HEAL / "manifest.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(manifest["clean_mix_share"], 0.30)
        self.assertLessEqual(manifest["clean_mix_share"], 0.50)

    def test_poisoned_set_flips_at_most_a_third(self):
        poisoned = load("dpo_pairs_poisoned.jsonl")
        if poisoned is None:
            self.skipTest("no poisoned set")
        flipped = sum(
            1 for pair in poisoned if pair["provenance"].get("poisoned")
        )
        self.assertGreater(flipped, 0)
        self.assertLessEqual(flipped / len(poisoned), 0.35)

    def test_evaluation_seeds_are_known_to_the_leak_guard(self):
        from heal.examples import evaluation_seeds

        reserved = evaluation_seeds()
        self.assertTrue(reserved)
        # the gate bucket and the sealed set must both be protected
        self.assertIn(6000, reserved)
        self.assertIn(20000, reserved)

    def test_no_sealed_seed_leaks_into_the_heal_set(self):
        sealed = {
            json.loads(line)["seed"]
            for line in (ROOT / "data" / "final_sealed" / "sealed_traces.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        for pair in self.pairs:
            self.assertNotIn(pair["provenance"]["seed"], sealed)


if __name__ == "__main__":
    unittest.main()
