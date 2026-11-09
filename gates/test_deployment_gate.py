import unittest

from gates.deployment_gate import evaluate_deployment_gate


BUCKETS = ("drift_targeted", "gate_regression")


def make_paired_records(count=40):
    baseline = []
    candidate = []

    base_success_count = int(count * 0.70)
    candidate_success_count = min(
        count,
        base_success_count + max(1, int(count * 0.20)),
    )

    for bucket in BUCKETS:
        for i in range(count):
            base_success = i < base_success_count
            candidate_success = i < candidate_success_count

            common = {
                "id": f"{bucket}_{i}",
                "split": bucket,
                "num_steps": 4,
                "total_tool_calls": 10,
                "redundant_calls": 1,
                "family": ("F1", "F2", "F4")[i % 3],
            }

            baseline.append({
                **common,
                "success": base_success,
                "failure_mode": (
                    None if base_success else "schema_violation"
                ),
            })

            candidate.append({
                **common,
                "success": candidate_success,
                "failure_mode": (
                    None if candidate_success else "schema_violation"
                ),
            })

    return baseline, candidate


class TestDeploymentGate(unittest.TestCase):

    def test_valid_candidate_passes_both_buckets(self):
        baseline, candidate = make_paired_records()

        result = evaluate_deployment_gate(baseline, candidate)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["passed"])

        for bucket in BUCKETS:
            self.assertEqual(
                result["buckets"][bucket]["status"], "PASS"
            )

    def test_both_buckets_are_required(self):
        baseline, candidate = make_paired_records()

        baseline = [
            r for r in baseline if r["split"] != "gate_regression"
        ]
        candidate = [
            r for r in candidate if r["split"] != "gate_regression"
        ]

        result = evaluate_deployment_gate(baseline, candidate)

        self.assertEqual(result["status"], "BLOCKED")

    def test_insufficient_bucket_samples_block(self):
        baseline, candidate = make_paired_records(count=20)

        result = evaluate_deployment_gate(baseline, candidate)

        for bucket in BUCKETS:
            self.assertEqual(
                result["buckets"][bucket]["checks"]
                ["minimum_bucket_samples"]["status"],
                "BLOCKED",
            )

    def test_small_families_are_skipped(self):
        baseline, candidate = make_paired_records(count=20)

        result = evaluate_deployment_gate(baseline, candidate)

        for bucket in BUCKETS:
            family_check = result["buckets"][bucket]["checks"][
                "worst_family_regression"
            ]
            self.assertEqual(family_check["status"], "PASS")
            self.assertTrue(
                family_check["value"]["skipped_families"]
            )

    def test_family_regression_fails(self):
        baseline, candidate = make_paired_records()

        for bucket in BUCKETS:
            records = [
                r for r in candidate
                if r["split"] == bucket and r["family"] == "F2"
            ]

            baseline_success_ids = {
                r["id"] for r in baseline
                if r["split"] == bucket and r["success"]
            }

            changed = 0
            for record in records:
                if record["id"] in baseline_success_ids:
                    record["success"] = False
                    record["failure_mode"] = "schema_violation"
                    changed += 1
                    if changed == 4:
                        break

        result = evaluate_deployment_gate(baseline, candidate)

        for bucket in BUCKETS:
            self.assertEqual(
                result["buckets"][bucket]["checks"]
                ["worst_family_regression"]["status"],
                "FAIL",
            )

    def test_steps_regression_fails(self):
        baseline, candidate = make_paired_records()

        for record in candidate:
            if record["success"]:
                record["num_steps"] = 5

        result = evaluate_deployment_gate(baseline, candidate)

        for bucket in BUCKETS:
            self.assertEqual(
                result["buckets"][bucket]["checks"]
                ["steps_on_success"]["status"],
                "FAIL",
            )

    def test_redundant_call_increase_fails(self):
        baseline, candidate = make_paired_records()

        for record in candidate:
            record["redundant_calls"] = 2

        result = evaluate_deployment_gate(baseline, candidate)

        for bucket in BUCKETS:
            self.assertEqual(
                result["buckets"][bucket]["checks"]
                ["redundant_call_rate"]["status"],
                "FAIL",
            )

    def test_new_failure_mode_above_two_percent_fails(self):
        baseline, candidate = make_paired_records()

        for bucket in BUCKETS:
            record = next(
                r for r in candidate
                if r["split"] == bucket and r["id"].endswith("_36")
            )
            record["failure_mode"] = "loop_collapse"

        result = evaluate_deployment_gate(baseline, candidate)

        for bucket in BUCKETS:
            check = result["buckets"][bucket]["checks"][
                "new_failure_mode_share"
            ]
            self.assertEqual(check["status"], "FAIL")
            self.assertEqual(check["value"]["denominator"], 40)

    def test_mismatched_task_ids_block_gate(self):
        baseline, candidate = make_paired_records()
        candidate[0]["id"] = "different_task"

        result = evaluate_deployment_gate(baseline, candidate)

        self.assertEqual(result["status"], "BLOCKED")

    def test_missing_family_blocks_family_check(self):
        baseline, candidate = make_paired_records()

        for records in (baseline, candidate):
            records[0].pop("family")

        result = evaluate_deployment_gate(baseline, candidate)

        self.assertEqual(
            result["buckets"]["drift_targeted"]["checks"]
            ["worst_family_regression"]["status"],
            "BLOCKED",
        )

        self.assertEqual(
            result["buckets"]["gate_regression"]["checks"]
            ["worst_family_regression"]["status"],
            "PASS",
        )


    def test_missing_failure_label_blocks_gate(self):
        baseline, candidate = make_paired_records()

        for records in (baseline, candidate):
            for record in records:
                if not record["success"]:
                    record.pop("failure_mode")
                    break

        result = evaluate_deployment_gate(baseline, candidate)

        self.assertEqual(
            result["buckets"]["drift_targeted"]["checks"]
            ["new_failure_mode_share"]["status"],
            "BLOCKED",
        )

        self.assertEqual(
            result["buckets"]["gate_regression"]["checks"]
            ["new_failure_mode_share"]["status"],
            "PASS",
        )

    def test_duplicate_task_ids_are_rejected(self):
        baseline, candidate = make_paired_records()
        baseline[1]["id"] = baseline[0]["id"]

        with self.assertRaises(ValueError):
            evaluate_deployment_gate(baseline, candidate)

    def test_gain_is_not_required_on_the_regression_bucket(self):
        """The plan gates the gain on the targeted bucket only.

        A candidate that fixes the drift and leaves clean tasks exactly as they
        were must deploy. Requiring a gain on the regression bucket rejects the
        normal, correct outcome.
        """
        baseline, candidate = make_paired_records(count=60)

        # Remove every gain on gate_regression.
        for base_record in baseline:
            if base_record["split"] != "gate_regression":
                continue
            match = next(
                record for record in candidate
                if record["id"] == base_record["id"]
            )
            match["success"] = base_record["success"]
            match["failure_mode"] = base_record["failure_mode"]

        result = evaluate_deployment_gate(baseline, candidate)

        regression = result["buckets"]["gate_regression"]
        self.assertEqual(
            regression["checks"]["task_success_gain"]["status"], "PASS"
        )
        self.assertEqual(
            regression["checks"]["task_success_gain"]["detail"], "not_required"
        )
        self.assertEqual(result["decision"], "DEPLOY")

    def test_efficiency_is_compared_on_the_clean_subset(self):
        """A baseline that is cheaper only because it failed must not win."""
        baseline, candidate = make_paired_records(count=60)

        for record in baseline + candidate:
            record["drift"] = "D2" if record["id"].endswith(("_0", "_1", "_2")) else "clean"

        # The candidate is slower on drifted tasks only: it now does the reads
        # the baseline skipped when it could not parse the drifted response.
        for record in candidate:
            if record["drift"] == "D2" and record["success"]:
                record["num_steps"] = 40

        result = evaluate_deployment_gate(baseline, candidate)

        for bucket in BUCKETS:
            check = result["buckets"][bucket]["checks"]["steps_on_success"]
            self.assertEqual(check["status"], "PASS")
            # no grounding field on these records, so the legacy definition runs
            # and says so rather than silently treating them as ungrounded
            self.assertIn("grounding not recorded", check["value"]["compared_on"])

    def test_efficiency_ignores_ungrounded_successes(self):
        """The check that produced the first verdict, and why it was wrong.

        A broken baseline answers `[]` on `none` tasks without reading anything:
        correct, cheap, ungrounded. Comparing all successes made the healed model
        look slower for doing the reads. Only grounded successes are comparable.
        """
        baseline, candidate = make_paired_records(count=60)

        for record in baseline + candidate:
            record["drift"] = "D1"
            record["grounded_success"] = record["success"]

        baseline_successes = [r for r in baseline if r["success"]]
        self.assertGreaterEqual(len(baseline_successes), 42)

        # 30 of the baseline's successes were cheap and ungrounded
        for record in baseline_successes[:30]:
            record["grounded_success"] = False
            record["num_steps"] = 2

        result = evaluate_deployment_gate(baseline, candidate)
        check = result["buckets"]["drift_targeted"]["checks"]["steps_on_success"]

        self.assertEqual(check["value"]["efficiency_rule"], "grounded-paired-v2")
        self.assertEqual(check["value"]["compared_on"], "grounded successes")
        self.assertEqual(check["value"]["paired_n"], 12)
        self.assertEqual(check["status"], "PASS")
        # the legacy number that rejected the candidate is still reported
        self.assertEqual(check["value"]["legacy_baseline_median"], 2)
        self.assertEqual(check["value"]["legacy_candidate_median"], 4)

    def test_no_common_grounded_success_blocks_the_comparison(self):
        baseline, candidate = make_paired_records(count=60)
        for record in baseline + candidate:
            record["drift"] = "D1"
            record["grounded_success"] = record["success"]
        for record in baseline:
            if record["success"]:
                record["grounded_success"] = False

        result = evaluate_deployment_gate(baseline, candidate)
        check = result["buckets"]["drift_targeted"]["checks"]["steps_on_success"]

        self.assertEqual(check["status"], "BLOCKED")
        self.assertIn("grounded", check["detail"])

    def test_efficiency_still_fails_on_clean_regression(self):
        baseline, candidate = make_paired_records(count=60)

        for record in baseline + candidate:
            record["drift"] = "clean"

        for record in candidate:
            if record["success"]:
                record["num_steps"] = 9

        result = evaluate_deployment_gate(baseline, candidate)

        for bucket in BUCKETS:
            self.assertEqual(
                result["buckets"][bucket]["checks"]["steps_on_success"]["status"],
                "FAIL",
            )


if __name__ == "__main__":
    unittest.main()