import unittest
from pathlib import Path

import numpy as np

from config_path.config_path import PROJECT_ROOT
from utility.utility_data.taxonomy import load_taxonomy


ROOT = Path(__file__).resolve().parents[1]


class TaxonomyTests(unittest.TestCase):
    def test_paths_are_portable_and_local(self):
        self.assertEqual(PROJECT_ROOT.resolve(), ROOT.resolve())

    def test_all_variants_load_deterministically_and_validate_hashes(self):
        for dataset in ("amazon-book", "yelp2018"):
            for policy in ("no_merge", "merge_t5", "merge_t10", "merge_t15"):
                first = load_taxonomy(dataset, policy)
                second = load_taxonomy(dataset, policy)
                self.assertEqual(first.taxonomy_hash, second.taxonomy_hash)
                np.testing.assert_array_equal(
                    first.item_to_leaf_id, second.item_to_leaf_id
                )
                np.testing.assert_array_equal(
                    first.valid_train_mask, second.valid_train_mask
                )

    def test_invalid_and_strict_cold_items_are_excluded_from_prototypes(self):
        for dataset in ("amazon-book", "yelp2018"):
            taxonomy = load_taxonomy(dataset, "no_merge")
            self.assertFalse(
                np.any(taxonomy.valid_train_mask & ~taxonomy.valid_taxonomy_mask)
            )
            self.assertFalse(
                np.any(taxonomy.valid_train_mask & ~taxonomy.train_observed_mask)
            )
            self.assertTrue(
                np.all(
                    taxonomy.item_to_leaf_id[taxonomy.valid_train_mask] >= 0
                )
            )

    def test_amazon_parent_granularity_is_deterministic_and_yelp_is_rejected(self):
        leaf = load_taxonomy("amazon-book", "no_merge", "leaf")
        parent_a = load_taxonomy("amazon-book", "no_merge", "parent")
        parent_b = load_taxonomy("amazon-book", "no_merge", "parent")
        self.assertEqual(parent_a.taxonomy_hash, parent_b.taxonomy_hash)
        self.assertNotEqual(leaf.taxonomy_hash, parent_a.taxonomy_hash)
        self.assertLess(parent_a.num_prototypes, leaf.num_prototypes)
        np.testing.assert_array_equal(
            parent_a.item_to_leaf_id, parent_b.item_to_leaf_id
        )
        with self.assertRaises(ValueError):
            load_taxonomy("yelp2018", "no_merge", "parent")
