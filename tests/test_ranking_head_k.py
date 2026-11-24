"""Per-user Recall/NDCG analyses rank the top items as the stored evaluation does (torch.topk), also when scores tie:
compute_batch_order_and_rank(head_k=...) puts the torch.topk selection and order first and keeps a full permutation."""

import unittest

import torch

from tools.ranking.inference import STORED_TOP_K, compute_batch_order_and_rank


class _Model:
    def __init__(self, rating):
        self.rating = rating

    def get_rating_for_test(self, users):
        return self.rating[users].clone()


class _Dataset:
    def __init__(self, seen):
        self.seen = seen

    def get_user_pos_items_for_evaluation(self, users, split):
        return [self.seen.get(int(u), []) for u in users]


class RankingHeadTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        rating = torch.rand(6, 200, generator=generator)
        rating[:, 10:40] = 1.0          # saturated sigmoid scores: thirty tied items per user
        self.model = _Model(rating)
        self.dataset = _Dataset({0: [12, 13], 3: [5]})
        self.users = list(range(6))

    def test_head_is_the_topk_selection_and_order(self):
        order, rank = compute_batch_order_and_rank(self.model, self.dataset, torch.device("cpu"), self.users,
                                                   head_k=STORED_TOP_K)
        rating = self.model.get_rating_for_test(torch.tensor(self.users))
        for row, items in enumerate(self.dataset.get_user_pos_items_for_evaluation(self.users, "test")):
            rating[row, items] = float("-inf")
        self.assertTrue(torch.equal(order[:, :STORED_TOP_K], torch.topk(rating, k=STORED_TOP_K).indices))

    def test_order_is_a_permutation_and_rank_its_inverse(self):
        order, rank = compute_batch_order_and_rank(self.model, self.dataset, torch.device("cpu"), self.users,
                                                   head_k=STORED_TOP_K)
        for row in range(len(self.users)):
            self.assertEqual(sorted(order[row].tolist()), list(range(200)))
            self.assertTrue(torch.equal(rank[row, order[row]], torch.arange(1, 201)))

    def test_seen_items_stay_below_every_candidate(self):
        order, _ = compute_batch_order_and_rank(self.model, self.dataset, torch.device("cpu"), self.users,
                                                head_k=STORED_TOP_K)
        self.assertEqual(set(order[0, -2:].tolist()), {12, 13})
        self.assertEqual(order[3, -1].item(), 5)

    def test_without_head_k_the_order_is_the_full_sort(self):
        order, _ = compute_batch_order_and_rank(self.model, self.dataset, torch.device("cpu"), self.users)
        rating = self.model.get_rating_for_test(torch.tensor(self.users))
        for row, items in enumerate(self.dataset.get_user_pos_items_for_evaluation(self.users, "test")):
            rating[row, items] = float("-inf")
        self.assertTrue(torch.equal(order, torch.argsort(rating, dim=1, descending=True)))


if __name__ == "__main__":
    unittest.main()
