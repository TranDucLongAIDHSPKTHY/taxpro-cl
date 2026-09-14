import json
import tempfile
import unittest
from pathlib import Path

from tools.analysis.compile_ablation_sweep import collect
from utility.utility_train.group_evaluator import evaluate_rankings


class AblationCompilerTests(unittest.TestCase):
    def payload(self):
        targets = {0: [0]}
        result = evaluate_rankings(
            {0: list(range(20))},
            targets,
            {"near_cold": targets, "long_tail": targets, "warm": targets},
        )
        result["protocol"] = {
            "candidate_catalog": "full",
            "target_split": "test",
        }
        return result

    def write_run(self, root, payload):
        directory = Path(root) / "amazon-book" / "main-config" / "seed42"
        directory.mkdir(parents=True)
        manifest = {
            "status": "completed",
            "dataset": "amazon-book",
            "training_seed": 42,
            "configuration": {
                "taxonomy_policy": "no_merge",
                "taxonomy_granularity": "leaf",
                "epsilon_max": 0.1,
                "temperature": 0.2,
                "augmentation_direction": "taxonomy",
            },
        }
        (directory / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (directory / "final_test_group_metrics.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_collects_24_metrics_and_nine_support_counts_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_run(directory, self.payload())
            rows, supports, incomplete = collect(Path(directory))
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            {(row["metric"], row["k"]) for row in rows},
            {
                ("precision", 10), ("precision", 20),
                ("recall", 10), ("recall", 20),
                ("ndcg", 10), ("ndcg", 20),
            },
        )
        self.assertEqual(len(supports), 3)
        self.assertEqual(incomplete, [])
        for row in supports:
            self.assertIn("eligible_users", row)
            self.assertIn("relevant_items", row)
            self.assertIn("positive_interactions", row)

    def test_rejects_completed_artifact_missing_precision_or_support(self):
        payload = self.payload()
        del payload["long_tail"]["precision"]["20"]
        with tempfile.TemporaryDirectory() as directory:
            self.write_run(directory, payload)
            rows, supports, incomplete = collect(Path(directory))
        self.assertEqual(rows, [])
        self.assertEqual(supports, [])
        self.assertIn("missing_metric:long_tail:precision@20", incomplete[0]["reason"])
