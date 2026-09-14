import unittest

import torch
from torch.nn import functional as F

from utility.utility_function.taxonomy_contrastive import create_item_views


class TaxonomyAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.items = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
        self.leaves = torch.tensor([0, 1, -1])
        self.valid = torch.tensor([True, True, False])
        self.prototypes = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    def test_taxonomy_displacement_direction_cosine_exceeds_point_999(self):
        view_a, _, details = create_item_views(
            self.items,
            self.leaves,
            self.valid,
            self.prototypes,
            epsilon_max=0.1,
        )
        displacement = view_a[:2] - self.items[:2]
        expected = self.prototypes[self.leaves[:2]] - self.items[:2]
        cosine = F.cosine_similarity(displacement, expected)
        self.assertTrue(torch.all(cosine > 0.999))
        self.assertTrue(torch.isfinite(details["direction"]).all())

    def test_displacement_is_bounded_and_invalid_item_is_unchanged(self):
        view_a, view_b, _ = create_item_views(
            self.items,
            self.leaves,
            self.valid,
            self.prototypes,
            epsilon_max=0.1,
        )
        self.assertLessEqual(float((view_a - self.items).norm(dim=1).max()), 0.100001)
        self.assertLessEqual(float((view_b - self.items).norm(dim=1).max()), 0.100001)
        torch.testing.assert_close(view_a[2], self.items[2])
        torch.testing.assert_close(view_b[2], self.items[2])

    def test_two_views_sample_independent_item_level_epsilons(self):
        generator = torch.Generator().manual_seed(42)
        _, _, details = create_item_views(
            self.items,
            self.leaves,
            self.valid,
            self.prototypes,
            generator=generator,
        )
        self.assertFalse(torch.equal(details["epsilon_a"], details["epsilon_b"]))
        self.assertGreater(torch.unique(details["epsilon_a"]).numel(), 1)

    def test_zero_distance_is_finite(self):
        items = torch.tensor([[1.0, 0.0]])
        view_a, view_b, details = create_item_views(
            items,
            torch.tensor([0]),
            torch.tensor([True]),
            torch.tensor([[1.0, 0.0]]),
            delta=1e-8,
        )
        self.assertTrue(torch.isfinite(view_a).all())
        self.assertTrue(torch.isfinite(view_b).all())
        self.assertTrue(torch.isfinite(details["direction"]).all())
        torch.testing.assert_close(view_a, items)

    def test_random_control_changes_only_direction_mechanism(self):
        generator = torch.Generator().manual_seed(7)
        tax_a, tax_b, tax_details = create_item_views(
            self.items[:2],
            self.leaves[:2],
            self.valid[:2],
            self.prototypes,
            generator=generator,
            direction_mode="taxonomy",
        )
        generator = torch.Generator().manual_seed(7)
        random_a, random_b, random_details = create_item_views(
            self.items[:2],
            self.leaves[:2],
            self.valid[:2],
            self.prototypes,
            generator=generator,
            direction_mode="random",
        )
        torch.testing.assert_close(
            tax_details["epsilon_a"], random_details["epsilon_a"]
        )
        torch.testing.assert_close(
            tax_details["epsilon_b"], random_details["epsilon_b"]
        )
        self.assertEqual(tax_a.shape, random_a.shape)
        self.assertEqual(tax_b.shape, random_b.shape)
        self.assertFalse(torch.equal(tax_details["direction"], random_details["direction"]))

    def test_invalid_augmentation_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            create_item_views(
                self.items, self.leaves, self.valid, self.prototypes,
                epsilon_max=-0.1,
            )
        with self.assertRaises(ValueError):
            create_item_views(
                self.items, self.leaves, self.valid, self.prototypes,
                delta=0.0,
            )
