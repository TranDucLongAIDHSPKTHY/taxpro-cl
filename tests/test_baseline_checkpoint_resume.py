import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from utility.utility_train.batch_test import _save_checkpoint
from utility.utility_train.trainer import (
    load_resume_checkpoint,
    prepare_resumed_training,
)


class _Dataset:
    def __init__(self, output_dir):
        self.training_output_dir = Path(output_dir)
        self.validation_history = []


class BaselineCheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "dataset": "amazon-book",
            "top_K": "[10, 20]",
            "selection_K": "20",
            "training_epochs": "1000",
            "early_stopping": "10",
            "interval": "10",
            "embedding_size": "2",
            "learn_rate": "0.001",
            "resume_checkpoint": "none",
        }

    def test_restores_model_optimizer_epoch_and_training_state(self):
        model = torch.nn.Linear(2, 1)
        model.training_seed = 42
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        state = {
            "count": 3,
            "epoch": 560,
            "recall": [0.1, 0.2],
            "ndcg": [0.05, 0.1],
            "stop": 0,
            "primary_metric": "recall@20",
            "primary_value": 0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_model.pt"
            _save_checkpoint(
                path, model, optimizer, 570, {"recall": [0.1, 0.2]},
                self.config, "recall@20", 0.2, state,
            )
            restored = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
            checkpoint = load_resume_checkpoint(
                path, restored, restored_optimizer, self.config,
                torch.device("cpu"), training_seed=42,
            )

        self.assertEqual(checkpoint["epoch"], 570)
        self.assertTrue(restored_optimizer.state_dict()["state"])
        for name, value in restored.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]))

    def test_prepare_resume_preserves_validation_history(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            model = torch.nn.Linear(2, 1)
            model.training_seed = 42
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            path = output_dir / "last_model.pt"
            state = {
                "count": 3, "epoch": 560, "recall": [0.1, 0.2],
                "ndcg": [0.05, 0.1], "stop": 0,
                "primary_metric": "recall@20", "primary_value": 0.2,
            }
            _save_checkpoint(
                path, model, optimizer, 570, {"recall": [0.1, 0.2]},
                self.config, "recall@20", 0.2, state,
            )
            history = [{"epoch": 570, "primary_value": 0.2}]
            (output_dir / "validation_metrics.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            resumed_config = dict(self.config, resume_checkpoint=str(path))
            dataset = _Dataset(output_dir)
            resumed_model = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.001)
            start_epoch, best = prepare_resumed_training(
                resumed_model, resumed_optimizer,
                SimpleNamespace(seed=42), resumed_config, dataset,
                torch.device("cpu"), logging.getLogger(__name__),
            )

        self.assertEqual(start_epoch, 570)
        self.assertEqual(best["count"], 3)
        self.assertEqual(dataset.validation_history, history)

    def test_rejects_incompatible_configuration_and_seed(self):
        model = torch.nn.Linear(2, 1)
        model.training_seed = 42
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_model.pt"
            _save_checkpoint(
                path, model, optimizer, 10, {}, self.config,
                "recall@20", 0.1, {},
            )
            with self.assertRaisesRegex(ValueError, "embedding_size"):
                load_resume_checkpoint(
                    path, torch.nn.Linear(2, 1), None,
                    dict(self.config, embedding_size="64"), torch.device("cpu"), 42,
                )
            with self.assertRaisesRegex(ValueError, "seed"):
                load_resume_checkpoint(
                    path, torch.nn.Linear(2, 1), None,
                    self.config, torch.device("cpu"), 0,
                )


if __name__ == "__main__":
    unittest.main()
