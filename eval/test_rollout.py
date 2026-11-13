"""Rollout harness tests. CPU only, no model, no network.

The harness is exercised with `ScriptedBackend`, which returns canned text in
place of a generation. Everything else is the real code path: the real
environment, the real drift wrappers, the real parser, the real oracle.
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from eval.rollout import (
    PROMPT_VARIANTS,
    ScriptedBackend,
    load_tasks,
    run_episode,
    summarise,
    to_record,
)
from expert.scripted import family_success as expert_family_success
from expert.scripted import run_expert
from oracle.checker import check_vulnerabilities
from world.snapshot import generate_snapshot


def call_text(name, arguments):
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


def a_vulnerable_seed(start=0, limit=400):
    for seed in range(start, start + limit):
        if check_vulnerabilities(generate_snapshot(seed)):
            return seed
    raise AssertionError("no vulnerable seed found")


def a_none_seed(start=0, limit=400):
    for seed in range(start, start + limit):
        if not check_vulnerabilities(generate_snapshot(seed)):
            return seed
    raise AssertionError("no none seed found")


def a_seed_with_a_safe_bucket(start=0, limit=400):
    """A vulnerable task that also has a bucket NOT in the answer.

    Needed to build a grounded-but-wrong answer: it has to name a real bucket,
    just the wrong one, or the label would be `ungrounded_answer` instead.
    """
    for seed in range(start, start + limit):
        snapshot = generate_snapshot(seed)
        truth = set(check_vulnerabilities(snapshot))
        if any(bucket["id"] not in truth for bucket in snapshot["buckets"]):
            return seed, snapshot, list(truth)
    raise AssertionError("no suitable seed found")


class TestRollout(unittest.TestCase):

    def test_expert_sequence_is_reproduced_as_a_success(self):
        seed = a_vulnerable_seed()
        expert = run_expert(generate_snapshot(seed), drift="D1")
        responses = [call_text(call["name"], call["arguments"]) for call in expert["tool_calls"]]

        trajectory = run_episode(
            ScriptedBackend(responses), seed, "D1", max_steps=30
        )

        self.assertTrue(trajectory["success"])
        self.assertTrue(trajectory["finished"])
        self.assertEqual(trajectory["invalid_calls"], 0)
        self.assertEqual(trajectory["redundant_calls"], 0)
        self.assertEqual(trajectory["final_answer"], check_vulnerabilities(generate_snapshot(seed)))
        self.assertEqual(
            trajectory["family_success"],
            expert_family_success(
                trajectory["final_answer"],
                check_vulnerabilities(generate_snapshot(seed)),
            ),
        )

    def test_prose_instead_of_a_call_is_a_schema_violation(self):
        seed = a_vulnerable_seed()
        backend = ScriptedBackend(["I cannot complete the review."])

        trajectory = run_episode(backend, seed, None, max_steps=4)

        self.assertFalse(trajectory["success"])
        self.assertFalse(trajectory["finished"])
        self.assertGreaterEqual(trajectory["invalid_calls"], 1)
        self.assertEqual(trajectory["total_tool_calls"], 0)

        record = to_record(trajectory, bucket="dev", tag="t", variant="clean")
        self.assertEqual(record["failure_mode"], "schema_violation")

    def test_one_nudge_is_given_before_giving_up(self):
        seed = a_vulnerable_seed()
        backend = ScriptedBackend(["no tools here"])

        run_episode(backend, seed, None, max_steps=4)

        # first generate, then one retry carrying the format reminder
        self.assertGreaterEqual(len(backend.seen_messages), 2)
        reminder = backend.seen_messages[1][-1]
        self.assertEqual(reminder["role"], "tool")
        self.assertIn("schema_violation", reminder["content"])

    def test_always_none_succeeds_only_on_none_tasks(self):
        none_seed = a_none_seed()
        vulnerable_seed = a_vulnerable_seed()
        responses = [call_text("finish", {"answer": []})]

        none_trajectory = run_episode(ScriptedBackend(responses), none_seed, None)
        vulnerable_trajectory = run_episode(
            ScriptedBackend(responses), vulnerable_seed, None
        )

        self.assertTrue(none_trajectory["success"])
        self.assertFalse(vulnerable_trajectory["success"])

    def test_loop_with_a_grounded_wrong_answer_is_labelled(self):
        seed, snapshot, truth = a_seed_with_a_safe_bucket()
        wrong = [bucket["id"] for bucket in snapshot["buckets"] if bucket["id"] not in truth][0]

        responses = (
            [call_text("list_buckets", {})] * 3
            + [call_text("finish", {"answer": [wrong]})]
        )
        trajectory = run_episode(
            ScriptedBackend(responses), seed, None, max_steps=8
        )
        record = to_record(trajectory, bucket="dev", tag="t", variant="clean")

        self.assertFalse(record["success"])
        self.assertGreater(record["redundant_calls"], 0)
        self.assertEqual(record["failure_mode"], "loop_collapse")

    def test_prompt_variant_reaches_the_system_message(self):
        seed = a_none_seed()
        backend = ScriptedBackend([call_text("finish", {"answer": []})])

        run_episode(
            backend,
            seed,
            None,
            prompt_suffix=PROMPT_VARIANTS["prompt_only_d1"],
        )

        system = backend.seen_messages[0][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("bucket_name", system["content"])

    def test_clean_prompt_has_no_hint(self):
        seed = a_none_seed()
        backend = ScriptedBackend([call_text("finish", {"answer": []})])

        run_episode(backend, seed, None, prompt_suffix=PROMPT_VARIANTS["clean"])

        system = backend.seen_messages[0][0]["content"]
        self.assertNotIn("bucket_name", system)
        self.assertNotIn("Note:", system)

    def test_max_steps_is_enforced(self):
        seed = a_vulnerable_seed()
        backend = ScriptedBackend([call_text("list_buckets", {})])

        trajectory = run_episode(backend, seed, None, max_steps=3)

        self.assertEqual(trajectory["num_steps"], 3)
        self.assertFalse(trajectory["finished"])

    def test_determinism_for_the_same_script(self):
        seed = a_vulnerable_seed()
        expert = run_expert(generate_snapshot(seed), drift="D2")
        responses = [call_text(call["name"], call["arguments"]) for call in expert["tool_calls"]]

        first = run_episode(ScriptedBackend(responses), seed, "D2", max_steps=30)
        second = run_episode(ScriptedBackend(responses), seed, "D2", max_steps=30)

        self.assertEqual(first["messages"], second["messages"])
        self.assertEqual(first["final_answer"], second["final_answer"])


class TestRecordContract(unittest.TestCase):

    def setUp(self):
        seed = a_vulnerable_seed()
        self.bucket = "drift_targeted"
        self.trajectory = run_episode(
            ScriptedBackend([call_text("finish", {"answer": []})]),
            seed,
            "D1",
        )
        self.record = to_record(
            self.trajectory, bucket=self.bucket, tag="v0", variant="clean"
        )

    def test_gate_fields_are_present(self):
        required = {
            "id",
            "bucket",
            "success",
            "num_steps",
            "total_tool_calls",
            "redundant_calls",
            "family_success",
            "failure_mode",
            "is_none_task",
        }
        self.assertFalse(required - set(self.record))

    def test_family_success_covers_all_three_families(self):
        self.assertEqual(set(self.record["family_success"]), {"F1", "F2", "F4"})

    def test_provenance_is_recorded(self):
        self.assertEqual(self.record["policy"], "v0")
        self.assertEqual(self.record["prompt_variant"], "clean")
        self.assertEqual(self.record["bucket"], self.bucket)

    def test_summary_balances_none_and_vulnerable(self):
        none_seed = a_none_seed()
        vulnerable_seed = a_vulnerable_seed()

        none_record = to_record(
            run_episode(
                ScriptedBackend([call_text("finish", {"answer": []})]), none_seed, None
            ),
            bucket="dev", tag="t", variant="clean",
        )
        vulnerable_record = to_record(
            run_episode(
                ScriptedBackend([call_text("finish", {"answer": []})]),
                vulnerable_seed,
                None,
            ),
            bucket="dev", tag="t", variant="clean",
        )

        summary = summarise([none_record, vulnerable_record])

        self.assertEqual(summary["task_success"], 0.5)
        self.assertEqual(summary["success_on_none"], 1.0)
        self.assertEqual(summary["success_on_vulnerable"], 0.0)
        self.assertEqual(summary["balanced_success"], 0.5)


class TestTaskLoading(unittest.TestCase):

    def test_sealed_requires_an_explicit_flag(self):
        with self.assertRaises(SystemExit):
            load_tasks("final_sealed", allow_sealed=False)

    def test_bucket_tasks_carry_seed_and_drift(self):
        tasks = load_tasks("drift_targeted")
        self.assertTrue(tasks)
        for task in tasks:
            self.assertIsInstance(task["seed"], int)
        self.assertEqual({task["drift"] for task in tasks}, {"D1"})


class TestStaleSchema(unittest.TestCase):
    """The demo drift: a pinned tool registry against a moved API."""

    def test_stale_prompt_advertises_the_old_name(self):
        from eval.run_agent import run_stale

        seed = a_vulnerable_seed()
        calls = run_stale(seed, None)["tool_calls"]
        backend = ScriptedBackend([call_text(c["name"], c["arguments"]) for c in calls])

        fresh = run_episode(
            ScriptedBackend([call_text(c["name"], c["arguments"]) for c in calls]),
            seed, None, drift_override="D1", max_steps=30, stale_schema=False,
        )
        stale = run_episode(
            backend, seed, None, drift_override="D1", max_steps=30, stale_schema=True,
        )

        system_fresh = fresh["messages"][0]["content"]
        system_stale = stale["messages"][0]["content"]
        self.assertIn("bucket_name", system_fresh)
        self.assertNotIn("bucket_name", system_stale)
        self.assertIn("Args: bucket (str)", system_stale)

    def test_stale_schema_makes_the_old_call_fail(self):
        from eval.run_agent import run_stale

        seed = a_vulnerable_seed()
        calls = run_stale(seed, None)["tool_calls"]
        # the reads fail, so a model that trusts them cannot ground its answer
        calls = [c for c in calls if c["name"] != "finish"] + [
            {"name": "finish", "arguments": {"answer": []}}
        ]
        trajectory = run_episode(
            ScriptedBackend([call_text(c["name"], c["arguments"]) for c in calls]),
            seed, None, drift_override="D1", max_steps=30, stale_schema=True,
        )

        tool_text = " ".join(
            message["content"] for message in trajectory["messages"]
            if message["role"] == "tool"
        )
        self.assertIn("unexpected argument", tool_text)

        from classifier.rules import classify
        self.assertEqual(classify(trajectory)["primary"], "drift_mismatch")

    def test_stale_schema_is_a_no_op_on_clean_tasks(self):
        """No drift means nothing to be stale about; the prompt must not change."""
        from eval.run_agent import run_stale

        seed = a_vulnerable_seed()
        calls = run_stale(seed, None)["tool_calls"]
        responses = [call_text(c["name"], c["arguments"]) for c in calls]

        fresh = run_episode(ScriptedBackend(responses), seed, None, stale_schema=False)
        stale = run_episode(ScriptedBackend(responses), seed, None, stale_schema=True)

        self.assertEqual(
            fresh["messages"][0]["content"], stale["messages"][0]["content"]
        )

    def test_gate_regression_keeps_per_task_drifts(self):
        """The bucket must not be forced onto one drift.

        Forcing D1 there would relabel its 42 clean tasks as drifted and destroy
        the regression check, which is what --drift-override D1 would have done
        before this was per-task.
        """
        tasks = load_tasks("gate_regression")
        drifts = {task["drift"] for task in tasks}

        self.assertIn(None, drifts, "no clean tasks left in gate_regression")
        self.assertTrue(drifts - {None}, "no drifted tasks left in gate_regression")

    def test_the_flag_is_recorded(self):
        seed = a_vulnerable_seed()
        trajectory = run_episode(
            ScriptedBackend([call_text("finish", {"answer": []})]),
            seed, None, drift_override="D1", stale_schema=True,
        )
        record = to_record(
            trajectory, bucket="drift_targeted", tag="v0", variant="clean"
        )
        self.assertTrue(record["stale_schema"])


class TestHandoffToTheHealer(unittest.TestCase):
    """The rollout output must be usable as DPO training data.

    Regression test for a real bug: the pair builder used to require the
    failure's messages to be byte-identical to the expert's, which a real model
    never produces. It now rebuilds both sides from the call sequence.
    """

    def _stale_rollout(self, seed, drift="D1"):
        from eval.run_agent import run_stale

        calls = run_stale(seed, drift)["tool_calls"]
        backend = ScriptedBackend([call_text(c["name"], c["arguments"]) for c in calls])
        return run_episode(backend, seed, drift, max_steps=30)

    def _a_failing_seed(self, drift="D1"):
        from eval.run_agent import run_stale
        from heal.replay import replay

        for seed in range(30000, 30080):
            calls = run_stale(seed, drift)["tool_calls"]
            if not replay(generate_snapshot(seed), calls, drift)["correct"]:
                return seed
        raise AssertionError("no failing seed found")

    def test_a_real_rollout_yields_a_minimal_edit_pair(self):
        from heal.examples import build_pair

        seed = self._a_failing_seed()
        trajectory = self._stale_rollout(seed)

        pair = build_pair(trajectory, "D1")

        self.assertIsNotNone(pair)
        self.assertTrue(pair["provenance"]["chosen_perfect"])
        self.assertEqual(pair["provenance"]["label"], "drift_mismatch")
        self.assertEqual(pair["chosen"][0]["role"], "assistant")
        self.assertEqual(pair["rejected"][0]["role"], "assistant")
        self.assertNotEqual(pair["chosen"][0], pair["rejected"][0])
        # two messages per call, plus system and user
        self.assertEqual(len(pair["prompt"]) % 2, 0)

    def test_untidy_model_text_still_yields_a_pair(self):
        """Whitespace and prose around the call must not matter."""
        from heal.examples import build_pair

        seed = self._a_failing_seed()
        trajectory = self._stale_rollout(seed)

        for message in trajectory["messages"]:
            if message["role"] == "assistant":
                message["content"] = (
                    "  \n\n" + message["content"].replace(">", ">  ") + "\n\n"
                )

        pair = build_pair(trajectory, "D1")
        self.assertIsNotNone(pair)
        self.assertTrue(pair["provenance"]["chosen_perfect"])

    def test_the_pair_is_rebuilt_canonically(self):
        from heal.examples import build_pair
        from expert.scripted import render_call

        seed = self._a_failing_seed()
        pair = build_pair(self._stale_rollout(seed), "D1")

        rejected_call = pair["rejected"][0]["content"]
        self.assertTrue(rejected_call.startswith("<tool_call>\n"))
        self.assertTrue(rejected_call.endswith("\n</tool_call>"))
        self.assertEqual(rejected_call, render_call(
            json.loads(rejected_call.split("\n")[1])["name"],
            json.loads(rejected_call.split("\n")[1])["arguments"],
        ))

    def test_a_no_tool_call_rollout_yields_no_pair(self):
        from heal.examples import build_pair

        seed = self._a_failing_seed()
        prose = run_episode(
            ScriptedBackend(["I cannot complete the review."]), seed, "D1", max_steps=3
        )
        self.assertIsNone(build_pair(prose, "D1"))

    def test_a_prompt_hinted_rollout_yields_no_pair(self):
        """A prompt-only run starts somewhere else; never train on it."""
        from eval.run_agent import run_stale
        from heal.examples import build_pair

        seed = self._a_failing_seed()
        calls = run_stale(seed, "D1")["tool_calls"]
        backend = ScriptedBackend([call_text(c["name"], c["arguments"]) for c in calls])
        hinted = run_episode(
            backend,
            seed,
            "D1",
            prompt_suffix=PROMPT_VARIANTS["prompt_only_d1"],
            max_steps=30,
        )
        self.assertIsNone(build_pair(hinted, "D1"))


class TestDriftOverride(unittest.TestCase):
    """Stage 1.6: the same tasks, the model forced onto each API version."""

    def test_override_changes_the_api_the_model_faces(self):
        from eval.run_agent import run_stale

        seed = a_vulnerable_seed()
        calls = run_stale(seed, None)["tool_calls"]
        responses = [call_text(c["name"], c["arguments"]) for c in calls]

        clean = run_episode(ScriptedBackend(responses), seed, None, max_steps=30)
        forced = run_episode(
            ScriptedBackend(responses), seed, None, max_steps=30, drift_override="D1"
        )

        def tool_text(trajectory):
            return " ".join(
                message["content"]
                for message in trajectory["messages"]
                if message["role"] == "tool"
            )

        # The same actions hit a different API: D1 rejects the old argument name.
        self.assertNotIn("unexpected argument", tool_text(clean))
        self.assertIn("unexpected argument", tool_text(forced))
        self.assertEqual(forced["drift"], "D1")

    def test_override_clean_normalises_to_no_drift(self):
        seed = a_vulnerable_seed()
        trajectory = run_episode(
            ScriptedBackend([call_text("finish", {"answer": []})]),
            seed,
            "D1",
            drift_override="clean",
        )
        self.assertIsNone(trajectory["drift"])
        self.assertIsNotNone(trajectory["drift_override"])

    def test_override_is_recorded_on_the_record(self):
        seed = a_vulnerable_seed()
        trajectory = run_episode(
            ScriptedBackend([call_text("finish", {"answer": []})]),
            seed,
            None,
            drift_override="D3",
        )
        record = to_record(
            trajectory, bucket="dev", tag="v0", variant="clean"
        )
        self.assertEqual(record["drift"], "D3")

    def test_sweep_writes_one_file_per_condition(self):
        import shutil

        from eval.rollout import _output_paths, _run_variant

        base = ROOT / "artifacts" / "_test_rollout"
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)

        seed = a_vulnerable_seed()
        tasks = [{"seed": seed, "drift": None}]
        backend = ScriptedBackend([call_text("finish", {"answer": []})])

        try:
            for forced in ("clean", "D1", "D2"):
                _run_variant(
                    backend,
                    tasks,
                    "dev",
                    "sweep",
                    "clean",
                    "",
                    None,
                    3,
                    64,
                    base,
                    drift_override=forced,
                    filename_label=f"clean_{forced}",
                    records_dir=base,
                )

            for forced in ("clean", "D1", "D2"):
                path, _ = _output_paths(
                    "dev", "sweep", f"clean_{forced}", base, base
                )
                self.assertTrue(path.is_file(), path.name)

            path, _ = _output_paths("dev", "sweep", "clean_D1", base, base)
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["drift"], "D1")
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
