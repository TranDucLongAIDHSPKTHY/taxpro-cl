"""The tie-order sensitivity ranks on the pre-sigmoid scores, breaks exact ties by item index, and counts the users
whose sigmoid top 20 contains tied scores or a tie across positions 20 and 21 (tools/analysis/tie_order_sensitivity.py)."""

import unittest

import torch

import tools.analysis.tie_order_sensitivity as tos


class _Model:
    def __init__(self, logits):
        self.logits = logits
        self.activation = torch.nn.Sigmoid()

    def get_rating_for_test(self, users):
        return self.activation(self.logits[users].clone())


class _Dataset:
    def __init__(self, seen, run):
        self.seen = seen
        self.training_output_dir = run

    def get_user_pos_items_for_evaluation(self, users, split):
        return [self.seen.get(int(u), []) for u in users]


class TieOrderTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(4, 100, generator=generator)
        logits[0, 10:35] = 20.0 + torch.arange(25, dtype=torch.float32)   # sigmoid saturates: 25 tied items
        logits[1, 50:53] = 30.0                                             # exact logit tie: order by index
        self.model = _Model(logits)
        self.dataset = _Dataset({2: [7]}, tos.ROOT / "log" / "unit-test-run")
        self.users = list(range(4))
        tos.TIES.clear()

    def test_order_follows_the_logits_with_index_ties(self):
        order, rank = tos.logit_order_and_rank(self.model, self.dataset, torch.device("cpu"), self.users)
        self.assertEqual(order[0, :25].tolist(), list(range(34, 9, -1)))
        self.assertEqual(order[1, :3].tolist(), [50, 51, 52])
        self.assertEqual(order[2, -1].item(), 7)
        for row in range(4):
            self.assertTrue(torch.equal(rank[row, order[row]], torch.arange(1, 101)))
        self.assertIsInstance(self.model.activation, torch.nn.Sigmoid)

    def test_ties_of_the_sigmoid_scores_are_counted(self):
        tos.logit_order_and_rank(self.model, self.dataset, torch.device("cpu"), self.users)
        (rec,) = tos.TIES.values()
        self.assertEqual(rec["users"], {0, 1, 2, 3})
        self.assertIn(0, rec["tie_within_top20"])
        self.assertIn(0, rec["tie_at_20_21"])
        self.assertIn(1, rec["tie_within_top20"])
        self.assertNotIn(1, rec["tie_at_20_21"])
        self.assertIn(1, rec["logit_tie_within_top21"])


if __name__ == "__main__":
    unittest.main()
