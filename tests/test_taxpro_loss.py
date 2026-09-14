import unittest

import torch
from torch.nn import functional as F

from utility.utility_function.taxonomy_contrastive import item_level_info_nce


class TaxProLossTests(unittest.TestCase):
    def test_info_nce_uses_diagonal_labels_and_mean_reduction(self):
        view_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        view_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = item_level_info_nce(view_a, view_b, temperature=0.2)
        logits = F.normalize(view_a, dim=1) @ F.normalize(view_b, dim=1).T / 0.2
        expected = F.cross_entropy(logits, torch.tensor([0, 1]), reduction="mean")
        torch.testing.assert_close(loss, expected)

    def test_single_item_returns_explicit_finite_zero(self):
        view = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss = item_level_info_nce(view, view.clone(), temperature=0.2)
        self.assertEqual(float(loss.item()), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_symmetric_mode_is_optional_not_default(self):
        view_a = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        view_b = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        asymmetric = item_level_info_nce(view_a, view_b, symmetric=False)
        symmetric = item_level_info_nce(view_a, view_b, symmetric=True)
        self.assertTrue(torch.isfinite(asymmetric))
        self.assertTrue(torch.isfinite(symmetric))

    def test_total_loss_is_finite(self):
        bpr = torch.tensor(0.5)
        reg = torch.tensor(0.01)
        cl = item_level_info_nce(
            torch.eye(2), torch.eye(2), temperature=0.2
        )
        total = bpr + reg + 0.1 * cl
        self.assertTrue(torch.isfinite(total))

    def test_non_positive_temperature_is_rejected(self):
        with self.assertRaises(ValueError):
            item_level_info_nce(torch.eye(2), torch.eye(2), temperature=0.0)
