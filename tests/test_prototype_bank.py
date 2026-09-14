import unittest

import torch

from utility.utility_function.taxonomy_contrastive import PrototypeBank


class PrototypeBankTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = torch.tensor(
            [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [9.0, 9.0]]
        )
        self.leaves = torch.tensor([0, 0, 1, -1])
        self.valid = torch.tensor([True, True, True, False])

    def test_initialization_is_exact_train_valid_mean(self):
        bank = PrototypeBank(2, 2, mu=0.9)
        evidence = bank.initialize(
            self.embeddings, self.leaves, self.valid, epoch=10
        )
        torch.testing.assert_close(
            bank.prototypes, torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        )
        self.assertEqual(evidence["item_support"], 3)
        self.assertEqual(bank.support.tolist(), [2, 1])

    def test_prototypes_are_registered_buffers_without_gradient(self):
        bank = PrototypeBank(2, 2)
        self.assertIn("prototypes", dict(bank.named_buffers()))
        self.assertNotIn("prototypes", dict(bank.named_parameters()))
        self.assertFalse(bank.prototypes.requires_grad)

    def test_ema_formula_and_active_leaf_only(self):
        bank = PrototypeBank(2, 2, mu=0.9)
        bank.initialize(self.embeddings, self.leaves, self.valid, epoch=1)
        updated_embeddings = self.embeddings.clone()
        updated_embeddings[0] = torch.tensor([4.0, 0.0])
        old_leaf_one = bank.prototypes[1].clone()
        stats = bank.ema_update(
            updated_embeddings,
            torch.tensor([0, 0, 3]),
            self.leaves,
            self.valid,
        )
        torch.testing.assert_close(
            bank.prototypes[0], 0.9 * torch.tensor([2.0, 0.0]) + 0.1 * torch.tensor([4.0, 0.0])
        )
        torch.testing.assert_close(bank.prototypes[1], old_leaf_one)
        self.assertEqual(stats["leaf_update_count"], 1)
        self.assertEqual(stats["item_update_count"], 1)

    def test_one_hundred_ema_batches_remain_finite(self):
        bank = PrototypeBank(2, 2, mu=0.9)
        bank.initialize(self.embeddings, self.leaves, self.valid, epoch=1)
        for index in range(100):
            values = self.embeddings + (index / 1000.0)
            bank.ema_update(
                values,
                torch.tensor([0, 1, 2, 3]),
                self.leaves,
                self.valid,
            )
        self.assertTrue(torch.isfinite(bank.prototypes).all())

    def test_prototype_initialization_runs_once(self):
        bank = PrototypeBank(2, 2)
        bank.initialize(self.embeddings, self.leaves, self.valid, epoch=1)
        with self.assertRaises(RuntimeError):
            bank.initialize(self.embeddings, self.leaves, self.valid, epoch=2)
