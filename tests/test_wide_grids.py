"""The widened KGCL and NCL grids of Deviation 2 of docs/extended_evaluation_protocol.md: run plans, the
separate registry, the selection on the union of cells, and the single test opening."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.analysis.wide_grid_analysis import edges
from tools.experiments import run_comparators, run_extended, run_wide_grids
from tools.experiments.run_wide_grids import KGCL_PACKAGE, NCL_PACKAGE, SEARCHES, plan, runs_to_open


class WideGridPlanTests(unittest.TestCase):
    def test_each_package_matches_the_deviation_run_count(self):
        self.assertEqual(len(plan(KGCL_PACKAGE)), 18)
        self.assertEqual(len(plan(NCL_PACKAGE)), 12)

    def test_new_runs_do_not_reuse_a_locked_directory_or_identifier(self):
        locked = [item for package in run_extended.EXPECTED_RUNS if package != "smoke"
                  for item in run_extended.planned(package)]
        locked += [item for package in run_comparators.PACKAGES for item in run_comparators.plan(package)]
        new = plan(KGCL_PACKAGE) + plan(NCL_PACKAGE)
        self.assertFalse({j.run_dir for j in new} & {j.run_dir for j in locked})
        self.assertFalse({j.run_id for j in new} & {j.run_id for j in locked})

    def test_kgcl_cells_change_only_the_two_searched_parameters(self):
        jobs = plan(KGCL_PACKAGE)
        self.assertTrue(all(set(dict(j.config)) == {"temperature", "ssl_weight"} for j in jobs))
        self.assertEqual({dict(j.config)["temperature"] for j in jobs}, {"0.25", "0.3"})
        self.assertEqual({dict(j.config)["ssl_weight"] for j in jobs}, {"0.02", "0.06", "0.1"})
        self.assertTrue(all(j.run_dir.as_posix().startswith("log/p0/extended/office-products/KGCL/grid-temp0.")
                            for j in jobs))

    def test_ncl_cells_keep_the_reference_except_temperature_and_clusters(self):
        reference = run_extended.inherited_config(run_extended.FAMILIES["office-products"]["NCL-default"], "NCL", {})
        for item in plan(NCL_PACKAGE):
            changed = {k for k, v in dict(item.config).items() if reference.get(k) != v}
            self.assertTrue(changed <= {"temperature", "k"}, changed)
        self.assertEqual({j.cell for j in plan(NCL_PACKAGE)},
                         {"grid-temp0.02-k500", "grid-temp0.02-k1000", "grid-temp0.03-k500", "grid-temp0.03-k1000"})

    def test_new_runs_have_their_own_registry(self):
        self.assertEqual(run_extended.registry_path(KGCL_PACKAGE), run_extended.ROOT / run_extended.WIDE_REGISTRY)
        self.assertEqual(run_extended.registry_path(NCL_PACKAGE), run_extended.ROOT / run_extended.WIDE_REGISTRY)
        self.assertEqual(run_extended.registry_path("grid-ncl"), run_extended.ROOT / run_extended.REGISTRY)
        self.assertEqual(run_extended.registry_path("comparator-kgcl"), run_extended.ROOT / run_extended.REGISTRY)

    def test_union_lists_existing_cells_first(self):
        existing, new = SEARCHES["kgcl"]["cells"]()
        self.assertEqual((len(existing), len(new)), (12, 6))
        self.assertEqual(existing[-1], "temp0.2-ssl0.1")
        existing, new = SEARCHES["ncl"]["cells"]()
        self.assertEqual((len(existing), len(new)), (12, 4))
        self.assertEqual(SEARCHES["ncl"]["family"]("temp0.05-k2000"), "log/p0/confirmatory/office-products/NCL/default")
        self.assertEqual(SEARCHES["ncl"]["family"]("temp0.02-k500"), "log/p0/extended/office-products/NCL/grid-temp0.02-k500")


class WideGridSelectionTests(unittest.TestCase):
    def test_no_test_split_opens_when_the_selection_is_unchanged(self):
        selection = {"KGCL": {"rule_R": "temp0.2-ssl0.06", "rule_R_before": "temp0.2-ssl0.06"}}
        self.assertEqual(runs_to_open("kgcl", selection), [])

    def test_a_changed_selection_opens_the_three_runs_of_its_cell(self):
        selection = {"ncl": {"rule_R": "temp0.02-k500", "rule_R_before": "temp0.05-k1000"}}
        runs = runs_to_open("ncl", selection)
        self.assertEqual(len(runs), 3)
        self.assertTrue(all("/extended/office-products/NCL/grid-temp0.02-k500/" in run.as_posix() for run in runs))

    def test_test_split_stays_closed_without_a_committed_selection(self):
        absent = Path("results/extended/_absent.json")
        with mock.patch.dict(SEARCHES["kgcl"], {"selection": absent}):
            with self.assertRaises(SystemExit):
                run_wide_grids.open_test("kgcl", "cpu")

    def test_selection_refuses_unfinished_runs(self):
        with mock.patch.object(run_wide_grids, "run_status", return_value="running"):
            with self.assertRaises(SystemExit):
                run_wide_grids.union_cells("ncl")

    def test_edges_name_the_parameters_at_the_widened_limits(self):
        self.assertEqual(edges("kgcl", "temp0.3-ssl0.06"), ["temperature"])
        self.assertEqual(edges("kgcl", "temp0.2-ssl0.1"), ["ssl_weight"])
        self.assertEqual(edges("ncl", "temp0.02-k500"), ["temperature", "clusters"])
        self.assertEqual(edges("ncl", "temp0.05-k1000"), [])


class AmendedProtocolTests(unittest.TestCase):
    def test_an_amended_protocol_runs_only_with_its_recorded_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.md").write_text("frozen", encoding="utf-8")
            frozen = run_extended.sha256_path(root / "protocol.md")
            (root / "protocol.sha256").write_text(frozen, encoding="utf-8")
            with mock.patch.object(run_extended, "ROOT", root), \
                    mock.patch.object(run_extended, "PROTOCOL", Path("protocol.md")):
                run_extended.require_frozen_protocol()
                (root / "protocol.md").write_text("frozen plus section 14", encoding="utf-8")
                with self.assertRaises(SystemExit):
                    run_extended.require_frozen_protocol()
                amended = run_extended.sha256_path(root / "protocol.md")
                (root / "protocol_deviations.sha256").write_text(f"{frozen}  old\n{amended}  new\n", encoding="utf-8")
                run_extended.require_frozen_protocol()
                (root / "protocol.md").write_text("frozen", encoding="utf-8")
                with self.assertRaises(SystemExit):
                    run_extended.require_frozen_protocol()


if __name__ == "__main__":
    unittest.main()
