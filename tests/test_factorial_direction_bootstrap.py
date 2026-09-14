import unittest

import numpy as np

from tools.analysis.factorial_direction_bootstrap import (
    bootstrap,
    pool_diffs_across_seeds,
)


class PoolDiffsAcrossSeedsTests(unittest.TestCase):
    def test_gvhd_two_user_three_seed_worked_example(self):
        """GVHD review 2026-09-12, Appendix 1: user 1 has per-seed diffs
        [1, 1, 1], user 2 has [-1, -1, -1]. After pooling, there must be
        exactly two units: +1 and -1 -- not six. Sampling per (user, seed)
        row instead of per user changes the sampling unit and is the exact
        bug this test guards against.
        """
        pooled_by_user = {
            "user_1": [1.0, 1.0, 1.0],
            "user_2": [-1.0, -1.0, -1.0],
        }
        result = pool_diffs_across_seeds(pooled_by_user)
        self.assertEqual(len(result), 2)
        self.assertEqual(sorted(result), [-1.0, 1.0])

    def test_users_with_different_seed_counts_still_pool_to_one_value_each(self):
        """A user need not appear in every seed pair (e.g. missing/failed
        checkpoint for one seed); pooling must still yield exactly one
        value per user, the mean of whichever seeds they did appear in.
        """
        pooled_by_user = {
            "user_1": [2.0, 4.0],  # only 2 of 3 seeds
            "user_2": [0.0, 0.0, 6.0],
        }
        result = pool_diffs_across_seeds(pooled_by_user)
        self.assertEqual(len(result), 2)
        self.assertIn(3.0, result)  # mean of [2, 4]
        self.assertIn(2.0, result)  # mean of [0, 0, 6]

    def test_bootstrap_sample_size_matches_unique_users_not_user_seed_rows(self):
        """End-to-end: n_users reported by bootstrap() must equal the
        number of unique users, matching GVHD's own reconciliation numbers
        (e.g. Amazon-Book/Near-Cold = 13,238, not the 3x-inflated 39,714
        that a per-(user, seed)-row sampling bug previously produced).
        """
        pooled_by_user = {"user_{}".format(i): [1.0, 1.0, 1.0] for i in range(13238)}
        diffs = pool_diffs_across_seeds(pooled_by_user)
        result = bootstrap(diffs, n_boot=100, rng=np.random.default_rng(0))
        self.assertEqual(result["n_users"], 13238)


if __name__ == "__main__":
    unittest.main()
