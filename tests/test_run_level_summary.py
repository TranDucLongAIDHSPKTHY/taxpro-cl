"""Main paper Table 6: the run-level summary reproduces the locked analysis and uses the stability rule of the
extended evaluation protocol (Section 7)."""

import unittest

from tools.analysis.run_level_summary import analysis_rows, lightgcn_rows, row


class RunLevelSummaryTests(unittest.TestCase):
    def test_every_comparison_and_group_has_one_row(self):
        rows = analysis_rows() + lightgcn_rows()
        keys = {(r["dataset"], r["comparison"], r["group"]) for r in rows}
        self.assertEqual(len(rows), 37)
        self.assertEqual(len(keys), 37)

    def test_lightgcn_on_the_redrawn_office_splits(self):
        rows = {r["group"]: r for r in lightgcn_rows()}
        self.assertEqual((rows["near_cold"]["positive"], rows["near_cold"]["negative"]), (0, 9))
        self.assertEqual((rows["overall"]["positive"], rows["overall"]["negative"]), (1, 8))
        self.assertEqual(rows["overall"]["classification"], "stable negative")

    def test_the_rule_allows_one_run_against_the_sign_and_needs_every_split_mean(self):
        self.assertEqual(row("d", "c", "g", [[1, 1, -1], [1, 1, 1], [1, 1, 1]])["classification"], "stable positive")
        self.assertEqual(row("d", "c", "g", [[1, -1, -1], [1, 1, 1], [1, 1, 1]])["classification"], "unstable")
        self.assertEqual(row("d", "c", "g", [[-3, 1, 1], [1, 1, 1], [1, 1, 1]])["classification"], "unstable")


if __name__ == "__main__":
    unittest.main()
