import unittest
from pathlib import Path

from tools.experiments.run_taxpro import (
    FULL_TRAINING_EPOCHS,
    build_command,
    effective_configuration,
    output_directory,
    parse_args,
)


class TaxProRunnerTests(unittest.TestCase):
    def test_full_run_uses_complete_locked_p0_configuration(self):
        args = parse_args(["--dataset", "amazon-book", "--device", "cpu"])
        config = effective_configuration(args)
        self.assertEqual(config["training_epochs"], FULL_TRAINING_EPOCHS)
        self.assertEqual(config["training_epochs"], 200)
        self.assertEqual(config["warm_start_epochs"], 20)
        expected = {
            "batch_size",
            "delta",
            "early_stopping",
            "embedding_size",
            "GCN_layer",
            "interval",
            "learn_rate",
            "mu",
            "reg_lambda",
            "selection_K",
            "symmetric_info_nce",
            "temperature",
            "test_batch_size",
            "top_K",
            "unknown_taxonomy_policy",
        }
        self.assertTrue(expected.issubset(config))

        command = build_command(args, config, 42, Path("unused"))
        for key in expected:
            self.assertIn("--" + key, command)

    def test_smoke_run_keeps_short_warm_and_total_epochs(self):
        args = parse_args(["--dataset", "yelp2018", "--smoke"])
        config = effective_configuration(args)
        self.assertEqual(config["training_epochs"], 5)
        self.assertEqual(config["warm_start_epochs"], 1)

    def test_output_uses_canonical_p0_taxprocl_layout(self):
        args = parse_args([
            "--dataset", "amazon-book", "--config-id", "a3-eps0.05"
        ])
        directory = output_directory(args, effective_configuration(args), 42)
        self.assertEqual(
            directory,
            args.output_root
            / "p0" / "taxprocl" / "amazon-book" / "a3-eps0.05" / "seed42",
        )


if __name__ == "__main__":
    unittest.main()
