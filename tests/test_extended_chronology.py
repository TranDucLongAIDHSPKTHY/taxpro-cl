"""The chronology of the extended evaluation (Online Resource 1, Tables S50 and S50b): every selection is
committed before the first test opening of its search, and the protocol is published before the first run."""

import unittest

from tools.analysis.extended_chronology import (
    PUSHES, WIDE_RUNS, first_commit, package_rows, published, registry_rows, utc)

SELECTIONS = {
    "results/extended/selection_grid.json": ("grid-simgcl", "grid-complete"),
    "results/extended/selection_kgcl.json": ("comparator-kgcl",),
    "results/extended/selection_galore.json": ("comparator-galore", "pretrain-lightgcn"),
    "results/extended/selection_ncl.json": ("grid-ncl",),
    "results/extended/selection_kgcl_wide.json": ("comparator-kgcl-wide",),
    "results/extended/selection_ncl_wide.json": ("grid-ncl-wide",),
}


class ChronologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packages = {row["package"]: row for row in package_rows(registry_rows())}

    def test_every_selection_is_committed_before_its_first_test_opening(self):
        for path, packages in SELECTIONS.items():
            commit = first_commit(path)
            openings = [utc(self.packages[p]["first_opening_utc"]) for p in packages
                        if p in self.packages and self.packages[p]["first_opening_utc"]]
            if commit is None:
                self.assertEqual(openings, [], f"{path}: test opened without a committed selection")
                continue
            for opening in openings:
                self.assertLess(utc(commit[1]), opening, path)

    def test_protocol_is_published_before_the_first_run(self):
        freeze = first_commit("docs/extended_evaluation_protocol.sha256")
        push = published(freeze[0])
        first_run = min(utc(row["first_start_utc"]) for row in self.packages.values())
        self.assertLess(utc(push[0]), first_run)

    def test_deviation_two_is_published_before_its_first_run(self):
        record = first_commit("docs/extended_evaluation_protocol_deviations.sha256")
        push = published(record[0])
        self.assertIsNotNone(push)
        for package in ("comparator-kgcl-wide", "grid-ncl-wide"):
            if package in self.packages:
                self.assertLess(utc(push[0]), utc(self.packages[package]["first_start_utc"]))

    def test_deviation_two_selections_are_published_before_their_test_opening(self):
        for path, package in (("results/extended/selection_kgcl_wide.json", "comparator-kgcl-wide"),
                              ("results/extended/selection_ncl_wide.json", "grid-ncl-wide")):
            opening = self.packages.get(package, {}).get("first_opening_utc")
            if not opening:
                continue
            push = published(first_commit(path)[0])
            self.assertIsNotNone(push, path)
            self.assertLess(utc(push[0]), utc(opening), path)

    def test_deviation_two_run_count_matches_its_runner(self):
        from tools.experiments.run_wide_grids import EXPECTED_RUNS

        self.assertEqual(WIDE_RUNS, sum(EXPECTED_RUNS.values()))

    def test_pushes_are_in_time_order_and_chain_their_heads(self):
        moments = [utc(moment) for moment, _, _ in PUSHES]
        self.assertEqual(moments, sorted(moments))
        for (_, _, head), (_, before, _) in zip(PUSHES, PUSHES[1:]):
            self.assertEqual(head, before)


if __name__ == "__main__":
    unittest.main()
