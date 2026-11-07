"""Decision log and adapter registry tests.

Covers the plan's explicit demand: "Test rollback by deploying v1_good, rolling
back to v0, and checking the pointer."
"""

import json
import shutil
import unittest
from pathlib import Path

import gates.registry as registry

ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / "artifacts" / "_test_registry"
from gates.deployment_gate import evaluate_deployment_gate


def paired(count=40, gain=0.20):
    baseline, candidate = [], []
    base = int(count * 0.5)
    extra = int(count * gain)
    for bucket in ("drift_targeted", "gate_regression"):
        for index in range(count):
            common = {
                "id": f"{bucket}_{index}",
                "bucket": bucket,
                "num_steps": 4,
                "total_tool_calls": 10,
                "redundant_calls": 1,
                "family_success": {"F1": True, "F2": True, "F4": True},
            }
            baseline.append(
                {
                    **common,
                    "success": index < base,
                    "failure_mode": None if index < base else "drift_mismatch",
                }
            )
            candidate.append(
                {
                    **common,
                    "success": index < base + extra,
                    "failure_mode": None if index < base + extra else "drift_mismatch",
                }
            )
    return baseline, candidate


class TestDecisions(unittest.TestCase):

    def test_three_published_outcomes(self):
        baseline, candidate = paired()
        deploy = evaluate_deployment_gate(baseline, candidate)
        self.assertEqual(deploy["decision"], "DEPLOY")

        reject = evaluate_deployment_gate(baseline, baseline)
        self.assertEqual(reject["decision"], "REJECT")

        blocked = evaluate_deployment_gate(baseline[:10], candidate[:10])
        self.assertEqual(blocked["decision"], "NEED_MORE_DATA")

    def test_rejection_names_the_failed_checks(self):
        baseline, _ = paired()
        result = evaluate_deployment_gate(baseline, baseline)
        self.assertTrue(result["failed_checks"])
        self.assertIn("task_success_gain", " ".join(result["failed_checks"]))
        self.assertIn("task_success_gain", result["reason"])

    def test_every_number_is_logged(self):
        path = TMP_ROOT / "decision_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        if True:
            baseline, candidate = paired()
            result = evaluate_deployment_gate(baseline, candidate)
            registry.log_decision(
                result, baseline="v0", candidate="v1_good", path=path
            )

            records = registry.read_decisions(path)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["decision"], "DEPLOY")
            self.assertIn("thresholds", record)
            self.assertIn("drift_targeted", record["buckets"])
            self.assertIn(
                "task_success_gain",
                record["buckets"]["drift_targeted"]["checks"],
            )


class TestRegistry(unittest.TestCase):

    def setUp(self):
        self.root = TMP_ROOT / "registry"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        root = self.root
        registry.REGISTRY_DIR = root
        registry.REGISTRY_FILE = root / "registry.json"
        registry.ACTIVE_FILE = root / "ACTIVE"

        for name in ("v0", "v1_good"):
            directory = root / name
            directory.mkdir()
            (directory / "adapter_model.bin").write_bytes(name.encode() * 32)
            registry.register(name, directory, parent="v0" if name == "v1_good" else None)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_register_hashes_content(self):
        state = registry.show()
        self.assertNotEqual(
            state["adapters"]["v0"]["sha256"],
            state["adapters"]["v1_good"]["sha256"],
        )
        self.assertEqual(state["adapters"]["v1_good"]["parent"], "v0")

    def test_deploy_then_rollback_moves_the_pointer(self):
        registry.activate("v0", reason="baseline")
        registry.activate("v1_good", reason="gate: DEPLOY")
        self.assertEqual(registry.active(), "v1_good")

        registry.rollback()
        self.assertEqual(registry.active(), "v0")

        history = registry.show()["history"]
        self.assertEqual(
            [(entry["from"], entry["to"]) for entry in history],
            [(None, "v0"), ("v0", "v1_good"), ("v1_good", "v0")],
        )

    def test_rollback_without_history_fails_loudly(self):
        with self.assertRaises(RuntimeError):
            registry.rollback()

    def test_unknown_adapter_is_refused(self):
        with self.assertRaises(KeyError):
            registry.activate("v9_missing")


if __name__ == "__main__":
    unittest.main()
