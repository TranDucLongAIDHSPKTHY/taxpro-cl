import json
import tempfile
import unittest
from pathlib import Path

from tools.experiments.run_baselines import is_completed as baseline_is_completed
from tools.experiments.run_taxpro import is_completed as taxpro_is_completed
from utility.utility_train.group_evaluator import evaluate_rankings


class RunnerResultContractTests(unittest.TestCase):
    def valid_bundle(self, directory):
        directory = Path(directory)
        targets = {0: [0]}
        group = evaluate_rankings(
            {0: list(range(20))},
            targets,
            {"near_cold": targets, "long_tail": targets, "warm": targets},
        )
        group["protocol"] = {
            "candidate_catalog": "full",
            "target_split": "test",
        }
        (directory / "run_manifest.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )
        (directory / "final_test_metrics.json").write_text(
            json.dumps({"test_metrics": {}}), encoding="utf-8"
        )
        (directory / "final_test_group_metrics.json").write_text(
            json.dumps(group), encoding="utf-8"
        )
        return group

    def test_both_runners_accept_only_complete_metric_and_support_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            group = self.valid_bundle(directory)
            self.assertTrue(baseline_is_completed(Path(directory)))
            self.assertTrue(taxpro_is_completed(Path(directory)))

            del group["near_cold"]["precision"]["10"]
            (Path(directory) / "final_test_group_metrics.json").write_text(
                json.dumps(group), encoding="utf-8"
            )
            self.assertFalse(baseline_is_completed(Path(directory)))
            self.assertFalse(taxpro_is_completed(Path(directory)))

    def test_runner_rejects_missing_support_or_non_test_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            group = self.valid_bundle(directory)
            del group["warm"]["positive_interactions"]
            (Path(directory) / "final_test_group_metrics.json").write_text(
                json.dumps(group), encoding="utf-8"
            )
            self.assertFalse(baseline_is_completed(Path(directory)))

            group = self.valid_bundle(directory)
            group["protocol"]["target_split"] = "validation"
            (Path(directory) / "final_test_group_metrics.json").write_text(
                json.dumps(group), encoding="utf-8"
            )
            self.assertFalse(taxpro_is_completed(Path(directory)))
