"""Label and gradient-path controls, re-drawn splits, the extended runner, the single test
opening and the analyses of docs/extended_evaluation_protocol.md."""

import collections
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from tests.helpers import ToyDataset
from tests.test_taxpro_cl_improved import improved_config, patched_improved_dependencies
from tools.analysis.extended_analysis import (
    at_upper_edge,
    classify,
    mean_std,
    rank_measure_table,
    sign_test,
    starting_point_wording,
    wording,
)
from tools.analysis.mechanism_diagnostics import (
    PerturbationProbe,
    by_bin_and_layer,
    degree_masks,
    embedding_cosine,
    embedding_shift,
    jaccard,
    scaling_check,
)
from tools.data.build_office_products import RATIOS, per_user_split
from tools.data.build_redrawn_splits import check_splitter, group_shares, locked_parts, read_pairs, redrawn_id
from tools.data.build_splits import _write_interactions_atomic
from tools.experiments.evaluate_sealed_test import check_against_group_metrics
from utility.utility_data.taxonomy import (
    REDRAWN_SPLIT_BASES,
    REDRAWN_SPLIT_SEEDS,
    SUPPORTED_DATASETS,
    perturb_taxonomy_labels,
)
from utility.utility_train.trainer import restore_validation_history

ROOT = Path(__file__).resolve().parents[1]
TaxProCLImproved = importlib.import_module("models.TaxPro-CL").TaxProCLImproved


class LabelPerturbationTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(0)
        self.leaves = generator.integers(0, 40, size=4000)
        self.leaves[generator.choice(4000, size=400, replace=False)] = -1
        self.valid = self.leaves >= 0

    def test_shuffle_keeps_sizes_mask_and_coverage(self):
        labels, valid = perturb_taxonomy_labels(self.leaves, self.valid, "shuffle", 1.0, 1042)
        np.testing.assert_array_equal(valid, self.valid)
        np.testing.assert_array_equal(labels[~self.valid], self.leaves[~self.valid])
        self.assertEqual(collections.Counter(labels.tolist()), collections.Counter(self.leaves.tolist()))

    def test_shuffle_retains_labels_at_the_chance_rate(self):
        labels, _ = perturb_taxonomy_labels(self.leaves, self.valid, "shuffle", 1.0, 1042)
        sizes = np.bincount(self.leaves[self.valid])
        chance = float(((sizes / self.valid.sum()) ** 2).sum())
        retained = float((labels[self.valid] == self.leaves[self.valid]).mean())
        self.assertLess(abs(retained - chance), 0.01)

    def test_noise_changes_exactly_the_chosen_share_and_keeps_the_mask(self):
        labels, valid = perturb_taxonomy_labels(self.leaves, self.valid, "noise", 0.5, 2042)
        np.testing.assert_array_equal(valid, self.valid)
        self.assertEqual(collections.Counter(labels.tolist()), collections.Counter(self.leaves.tolist()))
        changed = float((labels[self.valid] != self.leaves[self.valid]).mean())
        self.assertLessEqual(changed, 0.5)
        self.assertGreater(changed, 0.45)

    def test_missing_removes_half_of_the_valid_labels(self):
        labels, valid = perturb_taxonomy_labels(self.leaves, self.valid, "missing", 0.5, 2042)
        self.assertEqual(int(valid.sum()), int(round(0.5 * self.valid.sum())))
        np.testing.assert_array_equal(labels >= 0, valid)
        np.testing.assert_array_equal(labels[valid], self.leaves[valid])

    def test_seed_decides_the_result_and_global_state_is_untouched(self):
        before = np.random.get_state()[1].copy()
        first, _ = perturb_taxonomy_labels(self.leaves, self.valid, "noise", 0.5, 7)
        repeat, _ = perturb_taxonomy_labels(self.leaves, self.valid, "noise", 0.5, 7)
        other, _ = perturb_taxonomy_labels(self.leaves, self.valid, "noise", 0.5, 8)
        np.testing.assert_array_equal(np.random.get_state()[1], before)
        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, other))

    def test_rejects_an_unknown_mode_and_a_bad_rate(self):
        with self.assertRaises(ValueError):
            perturb_taxonomy_labels(self.leaves, self.valid, "uniform", 1.0, 1)
        with self.assertRaises(ValueError):
            perturb_taxonomy_labels(self.leaves, self.valid, "noise", 0.0, 1)


class ModelControlTests(unittest.TestCase):
    def build(self, **overrides):
        with patched_improved_dependencies():
            return TaxProCLImproved(improved_config(**overrides), ToyDataset(), torch.device("cpu"))

    def direction_row(self, model):
        """Gradient of one perturbed coordinate with respect to its own item state."""
        users = torch.zeros(model.dataset.num_users, 4)
        items = torch.arange(1.0, 17.0).reshape(4, 4).requires_grad_(True)
        torch.manual_seed(0)
        output = model._perturb_items(torch.cat([users, items]), 1)
        output[model.dataset.num_users, 0].backward()
        return items.grad[0]

    def test_defaults_leave_labels_and_gradient_path_unchanged(self):
        model = self.build()
        self.assertEqual(model.item_to_leaf_id.tolist(), [0, 0, 1, -1])
        self.assertEqual(model.taxonomy_perturbation, "none")
        self.assertIsNone(model.taxonomy_label_sha256)
        self.assertFalse(model.direction_detach)
        self.assertFalse(torch.equal(self.direction_row(model), torch.tensor([1.0, 0.0, 0.0, 0.0])))

    def test_detached_direction_removes_only_the_direction_path(self):
        model = self.build(direction_detach="true")
        torch.testing.assert_close(self.direction_row(model), torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_detached_direction_keeps_the_same_perturbed_state(self):
        items = torch.arange(1.0, 17.0).reshape(4, 4)
        embedding = torch.cat([torch.zeros(3, 4), items])
        torch.manual_seed(0)
        plain = self.build()._perturb_items(embedding, 1)
        torch.manual_seed(0)
        detached = self.build(direction_detach="true")._perturb_items(embedding, 1)
        self.assertLess(float((plain - detached).abs().max()), 1e-6)

    def test_shuffle_changes_labels_inside_the_mask_only(self):
        seen = set()
        for seed in range(12):
            model = self.build(taxonomy_perturbation="shuffle", taxonomy_perturbation_seed=str(seed))
            leaves = model.item_to_leaf_id.tolist()
            self.assertEqual(leaves[3], -1)
            self.assertEqual(sorted(leaves[:3]), [0, 0, 1])
            self.assertIsNotNone(model.taxonomy_label_retained_share)
            self.assertEqual(len(model.taxonomy_label_sha256), 64)
            seen.add(tuple(leaves))
        self.assertGreater(len(seen), 1)

    def test_missing_labels_leave_the_valid_masks_of_the_model(self):
        model = self.build(taxonomy_perturbation="missing", taxonomy_perturbation_rate="0.5",
                           taxonomy_perturbation_seed="2042")
        self.assertEqual(int(model.valid_taxonomy_mask.sum()), 1)
        torch.testing.assert_close(model.item_to_leaf_id >= 0, model.valid_taxonomy_mask)
        self.assertFalse(bool((model.valid_train_mask & ~model.valid_taxonomy_mask).any()))

    def test_both_directions_receive_the_same_missing_labels(self):
        common = dict(taxonomy_perturbation="missing", taxonomy_perturbation_rate="0.5",
                      taxonomy_perturbation_seed="2042")
        prototype = self.build(**common)
        random_direction = self.build(augmentation_direction="random", **common)
        self.assertEqual(prototype.taxonomy_label_sha256, random_direction.taxonomy_label_sha256)
        torch.testing.assert_close(prototype.valid_train_mask, random_direction.valid_train_mask)

    def test_controls_are_recorded_and_checked_in_checkpoint_metadata(self):
        model = self.build(direction_detach="true")
        metadata = model.checkpoint_metadata()
        self.assertTrue(metadata["direction_detach"])
        with self.assertRaises(ValueError):
            model.validate_checkpoint_metadata({**metadata, "direction_detach": False})

    def test_checkpoint_with_other_labels_is_refused(self):
        model = self.build(taxonomy_perturbation="shuffle", taxonomy_perturbation_seed="1042")
        metadata = model.checkpoint_metadata()
        with self.assertRaises(ValueError):
            model.validate_checkpoint_metadata({**metadata, "taxonomy_label_sha256": "0" * 64})

    def test_old_checkpoints_without_the_new_keys_still_validate(self):
        model = self.build()
        metadata = model.checkpoint_metadata()
        for key in ("taxonomy_perturbation", "taxonomy_perturbation_rate", "taxonomy_perturbation_seed",
                    "taxonomy_label_sha256", "direction_detach"):
            metadata.pop(key)
        model.validate_checkpoint_metadata(metadata)

    def test_detached_direction_requires_the_prototype_direction(self):
        with self.assertRaises(ValueError):
            self.build(direction_detach="true", augmentation_direction="random")

    def test_rejects_an_unknown_perturbation(self):
        with self.assertRaises(ValueError):
            self.build(taxonomy_perturbation="uniform")


class RedrawnSplitTests(unittest.TestCase):
    def test_splitter_reproduces_every_locked_split(self):
        for base in REDRAWN_SPLIT_BASES:
            check_splitter(base, locked_parts(base))

    def test_written_file_matches_the_locked_bytes(self):
        base = REDRAWN_SPLIT_BASES[0]
        parts = locked_parts(base)
        pool = parts["train"] | parts["validation"] | parts["test"]
        train, _, _ = per_user_split(pool, 42, RATIOS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            _write_interactions_atomic(path, train)
            self.assertEqual(path.read_bytes(), (ROOT / "dataset_verify" / base / "train.txt").read_bytes())

    def test_new_seed_is_a_different_partition_of_the_same_pool(self):
        base = REDRAWN_SPLIT_BASES[0]
        parts = locked_parts(base)
        pool = parts["train"] | parts["validation"] | parts["test"]
        train, validation, test = per_user_split(pool, REDRAWN_SPLIT_SEEDS[0], RATIOS)
        self.assertEqual(train | validation | test, pool)
        self.assertEqual(len(train) + len(validation) + len(test), len(pool))
        self.assertNotEqual(test, parts["test"])

    def test_split_seeds_and_identifiers(self):
        self.assertEqual(REDRAWN_SPLIT_SEEDS, (101, 102, 103))
        self.assertIn(redrawn_id("office-products", 101), SUPPORTED_DATASETS)
        self.assertNotIn(redrawn_id("amazon-book", 101), SUPPORTED_DATASETS)
        self.assertIn("office-products", SUPPORTED_DATASETS)

    def test_read_pairs_keeps_every_interaction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.txt"
            path.write_text("0 1 2\n3 4\n", encoding="utf-8")
            self.assertEqual(read_pairs(path), {(0, 1), (0, 2), (3, 4)})

    def test_group_shares_count_observed_items(self):
        self.assertEqual(group_shares([0, 1, 5, 6, 11]), {"near_cold_percent": 50.0, "long_tail_percent": 75.0})


@unittest.skipUnless((ROOT / "log" / "p0" / "taxprocl").is_dir(), "needs the local run manifests")
class RunnerPlanTests(unittest.TestCase):
    def test_every_package_matches_the_protocol_run_count(self):
        from tools.experiments.run_extended import EXPECTED_RUNS, planned

        for package, count in EXPECTED_RUNS.items():
            self.assertEqual(len(planned(package)), count, package)
        self.assertEqual(sum(count for name, count in EXPECTED_RUNS.items() if name != "smoke"), 357)

    def test_pretraining_keeps_the_reference_except_the_depth_and_stays_closed(self):
        from tools.experiments.run_extended import CLOSED_PACKAGES, FAMILIES, inherited_config, planned

        reference = inherited_config(FAMILIES["office-products"]["LightGCN"], "LightGCN", {})
        jobs = planned("pretrain-lightgcn")
        self.assertEqual({j.seed for j in jobs}, {42, 0, 1})
        for item in jobs:
            self.assertEqual({k: v for k, v in dict(item.config).items() if reference.get(k) != v},
                             {"GCN_layer": "2", "pool_layer_zero": "false"})
        self.assertIn("log/p0/extended/office-products/LightGCN/pretrain-2layers/seed0",
                      {j.run_dir.as_posix() for j in jobs})
        self.assertIn("pretrain-lightgcn", CLOSED_PACKAGES)

    def test_controls_differ_from_the_reference_only_in_their_flags(self):
        from tools.experiments.run_extended import FAMILIES, inherited_config, planned

        reference = inherited_config(FAMILIES["amazon-book"]["V3"], "TaxPro-CL", {})
        item = next(j for j in planned("controls-amazon-book") if j.arm == "V3-shuffled-detached" and j.seed == 0)
        changed = {k: v for k, v in dict(item.config).items() if reference.get(k) != v}
        self.assertEqual(changed, {"taxonomy_perturbation": "shuffle", "taxonomy_perturbation_rate": "1.0",
                                   "taxonomy_perturbation_seed": "1000", "direction_detach": "true"})

    def test_label_seeds_follow_the_training_seed(self):
        from tools.experiments.run_extended import planned

        for item in planned("controls-office-products"):
            config = dict(item.config)
            if "taxonomy_perturbation" in config:
                self.assertEqual(int(config["taxonomy_perturbation_seed"]), 1000 + item.seed)
        for item in planned("label-quality-office-products"):
            self.assertEqual(int(dict(item.config)["taxonomy_perturbation_seed"]), 2000 + item.seed)

    def test_missing_arms_share_one_label_seed_per_training_seed(self):
        from tools.experiments.run_extended import planned

        jobs = [j for j in planned("label-quality-musical-instruments") if j.arm.endswith("missing50")]
        by_seed = collections.defaultdict(set)
        for item in jobs:
            by_seed[item.seed].add(dict(item.config)["taxonomy_perturbation_seed"])
        self.assertTrue(all(len(values) == 1 for values in by_seed.values()))

    def test_grid_covers_the_new_cells_only_with_the_fixed_gamma(self):
        from tools.experiments.run_extended import planned

        complete = planned("grid-complete")
        self.assertEqual({dict(j.config)["gamma_cold"] for j in complete}, {"1.5"})
        self.assertEqual({dict(j.config)["temperature_user"] for j in complete}, {"0.2", "0.3", "0.4"})
        self.assertEqual({dict(j.config)["temperature"] for j in complete}, {"0.15", "0.2", "0.25"})
        pairs = {(dict(j.config)["temperature"], dict(j.config)["temperature_user"]) for j in complete}
        self.assertEqual(len(pairs), 8)
        self.assertNotIn(("0.15", "0.2"), pairs)
        simgcl = planned("grid-simgcl")
        self.assertEqual({dict(j.config)["temperature"] for j in simgcl}, {"0.25", "0.3", "0.4"})
        self.assertEqual({dict(j.config)["epsilon"] for j in simgcl}, {"0.05", "0.1", "0.2"})

    def test_complete_grid_union_holds_the_locked_cell_once(self):
        from tools.experiments.run_extended import grid_family, search_cells

        cells = search_cells()["complete_configuration"]
        self.assertEqual(len(cells), 14)
        self.assertEqual(len(set(cells)), 14)
        self.assertNotIn("temp0.15-tempuser0.2", cells)
        self.assertEqual(grid_family("complete_configuration", "temp0.15-gamma1.5"),
                         "log/p0/confirmatory/office-products/TaxPro-CL/step2-no_merge-temp0.15-gamma1.5")
        self.assertEqual(len(search_cells()["simgcl"]), 21)

    def test_ncl_grid_reuses_the_default_cell_and_changes_only_its_two_parameters(self):
        from tools.experiments.run_extended import CLOSED_PACKAGES, FAMILIES, grid_family, inherited_config, \
            planned, search_cells

        reference = inherited_config(FAMILIES["office-products"]["NCL-default"], "NCL", {})
        jobs = planned("grid-ncl")
        self.assertEqual(len(jobs), 33)
        cells = {(dict(j.config)["temperature"], dict(j.config)["k"]) for j in jobs}
        self.assertEqual(len(cells), 11)
        self.assertNotIn(("0.05", "2000"), cells)
        for item in jobs:
            self.assertTrue(set(k for k, v in dict(item.config).items() if reference.get(k) != v)
                            <= {"temperature", "k"})
        self.assertEqual(len(search_cells()["ncl"]), 12)
        self.assertEqual(grid_family("ncl", "temp0.05-k2000"), "log/p0/confirmatory/office-products/NCL/default")
        self.assertEqual(grid_family("ncl", "temp0.1-k3000"), "log/p0/extended/office-products/NCL/grid-temp0.1-k3000")
        self.assertIn("grid-ncl", CLOSED_PACKAGES)

    def test_ncl_test_split_opens_only_for_a_new_selected_cell(self):
        from tools.experiments.run_extended import grid_runs_to_open

        self.assertEqual(grid_runs_to_open({"ncl": {"rule_R": "temp0.05-k2000"}}, ("ncl",)), [])
        runs = grid_runs_to_open({"ncl": {"rule_R": "temp0.15-k1000"}}, ("ncl",))
        self.assertEqual(len(runs), 3)
        self.assertTrue(all("/extended/office-products/NCL/grid-temp0.15-k1000/" in run for run in runs))

    def test_every_reference_run_is_registered_with_its_environment(self):
        from tools.experiments.run_extended import reference_rows

        rows = reference_rows()
        self.assertEqual(len(rows), len({row["run_id"] for row in rows}))
        self.assertTrue(all(row["exp"] == "reference" and row["test_eval_utc"] == "" for row in rows))
        book = {(row["arm"], row["train_seed"]): row["env"] for row in rows if row["dataset"] == "amazon-book"}
        for arm in ("V0", "V1", "V2", "V3"):
            for seed in (42, 0, 1):
                self.assertEqual(book[arm, seed], "A", (arm, seed))

    def test_grid_test_split_opens_only_for_selected_cells_still_sealed(self):
        from tools.experiments.run_extended import grid_runs_to_open

        selection = {"complete_configuration": {"rule_R": "temp0.15-gamma1.5"},
                     "simgcl": {"rule_R": "temp0.2-eps0.1"}}
        self.assertEqual(grid_runs_to_open(selection), [])
        selection["simgcl"]["rule_R"] = "temp0.3-eps0.1"
        runs = grid_runs_to_open(selection)
        self.assertEqual(len(runs), 3)
        self.assertTrue(all("/extended/office-products/SimGCL/grid-temp0.3-eps0.1/" in run for run in runs))
        selection["complete_configuration"]["rule_R"] = "temp0.2-tempuser0.3"
        self.assertTrue(any("/TaxPro-CL/grid-tempuser0.3-no_merge-temp0.2-gamma1.5/" in run
                            for run in grid_runs_to_open(selection)))

    def test_selection_refuses_unfinished_runs(self):
        from tools.experiments import run_extended

        absent = Path("results/extended/_absent.json")
        for name in ("grid", "ncl"):
            _, packages, sides = run_extended.SEARCHES[name]
            with mock.patch.dict(run_extended.SEARCHES, {name: (absent, packages, sides)}), \
                    mock.patch.object(run_extended, "run_status", return_value="running"):
                with self.assertRaises(SystemExit):
                    run_extended.freeze_selection(name)

    def test_test_split_stays_closed_without_a_committed_selection(self):
        from tools.experiments import run_extended

        absent = Path("results/extended/_absent.json")
        for name in ("grid", "ncl"):
            _, packages, sides = run_extended.SEARCHES[name]
            with mock.patch.dict(run_extended.SEARCHES, {name: (absent, packages, sides)}):
                with self.assertRaises(SystemExit):
                    run_extended.open_test(name, "cpu")

    def test_test_split_stays_closed_while_a_package_is_unfinished(self):
        from tools.experiments import run_extended

        with mock.patch.object(run_extended, "run_status", return_value="running"):
            with self.assertRaises(SystemExit):
                run_extended.open_test("controls-office-products", "cpu")

    def test_runs_need_a_protocol_that_matches_its_hash(self):
        from tools.experiments import run_extended

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.md").write_text("text", encoding="utf-8")
            with mock.patch.object(run_extended, "ROOT", root), \
                    mock.patch.object(run_extended, "PROTOCOL", Path("protocol.md")):
                with self.assertRaises(SystemExit):
                    run_extended.require_frozen_protocol()
                (root / "protocol.sha256").write_text("0" * 64, encoding="utf-8")
                with self.assertRaises(SystemExit):
                    run_extended.require_frozen_protocol()
                digest = run_extended.sha256_path(root / "protocol.md")
                (root / "protocol.sha256").write_text(digest, encoding="utf-8")
                run_extended.require_frozen_protocol()

    def test_run_families_follow_the_confirmatory_layout(self):
        from tools.experiments.run_extended import planned

        root = "log/p0/extended"
        def paths(package):
            return {j.run_dir.as_posix() for j in planned(package)}

        self.assertIn(f"{root}/office-products/SimGCL/grid-temp0.25-eps0.05/seed42", paths("grid-simgcl"))
        self.assertIn(f"{root}/office-products/TaxPro-CL/grid-tempuser0.3-no_merge-temp0.15-gamma1.5/seed42",
                      paths("grid-complete"))
        self.assertIn(f"{root}/office-products/NCL/grid-temp0.1-k1000/seed42", paths("grid-ncl"))
        self.assertIn(f"{root}/amazon-book/TaxPro-CL/control-V3-shuffled-merge_t10-temp0.1-gamma1.5/seed42",
                      paths("controls-amazon-book"))
        self.assertIn(f"{root}/amazon-book/SimGCL/repeat-tuned-temp0.15-eps0.1/seed3",
                      paths("seeds-amazon-book-simgcl"))
        self.assertIn(f"{root}/office-products-split101/LightGCN/split-default/seed1",
                      paths("splits-office-products-lightgcn"))
        self.assertTrue(all(path.startswith(f"{root}/smoke/") for path in paths("smoke")))

    def test_run_identifiers_are_unique_across_packages(self):
        from tools.experiments.run_extended import EXPECTED_RUNS, planned

        identifiers = [j.run_id for package in EXPECTED_RUNS if package != "smoke" for j in planned(package)]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertIn("controls_amazon-book_V3-shuffled_s42_r0", identifiers)

    def test_new_seed_runs_use_only_the_new_seeds(self):
        from tools.experiments.run_extended import planned

        seeds = {item.seed for package in ("seeds-amazon-book-factorial", "seeds-amazon-book-simgcl",
                                           "seeds-yelp2018") for item in planned(package)}
        self.assertEqual(seeds, {2, 3, 4})

    def test_redrawn_runs_keep_the_reference_configuration(self):
        from tools.experiments.run_extended import planned

        jobs = [j for j in planned("splits-office-products") if j.dataset == "office-products-split101"]
        complete = dict(next(j for j in jobs if j.cell == "split-V3-no_merge-temp0.15-gamma1.5").config)
        self.assertEqual(complete["temperature"], "0.15")
        self.assertEqual(complete["gamma_cold"], "1.5")
        self.assertEqual(complete["dataset"], "office-products-split101")


class ResumeHistoryTests(unittest.TestCase):
    def history(self, directory, epochs):
        rows = [{"epoch": epoch, "primary_value": 0.1} for epoch in epochs]
        (Path(directory) / "validation_metrics.json").write_text(json.dumps(rows), encoding="utf-8")

    def test_history_is_restored_up_to_the_resumed_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            self.history(directory, [1, 2, 3])
            dataset = SimpleNamespace(training_output_dir=directory, validation_history=[])
            restore_validation_history(dataset, 2)
            self.assertEqual([row["epoch"] for row in dataset.validation_history], [1, 2])

    def test_runner_refuses_a_history_with_gaps(self):
        from tools.experiments.run_extended import history_is_complete

        with tempfile.TemporaryDirectory() as directory:
            manifest = {"configuration": {"interval": "1"}}
            (Path(directory) / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.history(directory, [1, 2, 3])
            self.assertTrue(history_is_complete(Path(directory)))
            self.history(directory, [23, 24])
            self.assertFalse(history_is_complete(Path(directory)))


class SingleOpeningTests(unittest.TestCase):
    def test_per_user_values_must_reproduce_the_group_metrics(self):
        from tools.analysis.factorial_multiplicity import GROUPS

        arrays, stored = {}, {}
        for group in GROUPS:
            arrays[f"{group}_recall"] = np.asarray([0.0, 0.5])
            arrays[f"{group}_ndcg"] = np.asarray([0.25, 0.25])
            stored[group] = {"recall": {"20": 0.25}, "ndcg": {"20": 0.25}}
        check_against_group_metrics(arrays, stored)
        stored[GROUPS[0]]["recall"]["20"] = 0.3
        with self.assertRaises(RuntimeError):
            check_against_group_metrics(arrays, stored)

    def test_a_run_with_a_test_result_is_refused(self):
        from tools.experiments.evaluate_sealed_test import evaluate

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "final_test_metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                evaluate(Path(directory), torch.device("cpu"), None)


class ExternalTableTests(unittest.TestCase):
    def test_scores_stay_above_the_seen_item_marker_and_keep_the_order(self):
        from models.ExternalEmbeddings import ExternalEmbeddings

        dataset = SimpleNamespace(num_users=2, num_items=3)
        model = ExternalEmbeddings({"embedding_size": 2}, dataset, torch.device("cpu"))
        users = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
        items = torch.tensor([[2.0, 0.0], [1.5, 0.0], [-1.0, 0.0]])
        model.load_tables(users, items)
        ratings = model.get_rating_for_test(torch.tensor([0, 1]))
        self.assertTrue(bool((ratings > 0).all()))
        self.assertTrue(bool((ratings[0, 0] > ratings[0, 1]) and (ratings[0, 1] > ratings[0, 2])))

    def test_scores_match_the_sigmoid_of_the_repository_models(self):
        from models.ExternalEmbeddings import ExternalEmbeddings

        dataset = SimpleNamespace(num_users=1, num_items=2)
        model = ExternalEmbeddings({"embedding_size": 1}, dataset, torch.device("cpu"))
        model.load_tables(torch.tensor([[1.0]]), torch.tensor([[30.0], [40.0]]))
        ratings = model.get_rating_for_test(torch.tensor([0]))
        self.assertEqual(ratings.dtype, torch.float32)
        self.assertEqual(float(ratings[0, 0]), float(ratings[0, 1]))

    def test_lightgcn_can_leave_layer_zero_out_of_the_pooling(self):
        from models.LightGCN import LightGCN

        model = LightGCN.__new__(LightGCN)
        torch.nn.Module.__init__(model)
        model.config = {"GCN_layer": 2}
        model.dataset = SimpleNamespace(num_users=1, num_items=1)
        model.user_embedding = torch.nn.Embedding(1, 1)
        model.item_embedding = torch.nn.Embedding(1, 1)
        with torch.no_grad():
            model.user_embedding.weight.fill_(1.0)
            model.item_embedding.weight.fill_(1.0)
        model.Graph = (2.0 * torch.eye(2)).to_sparse()
        model.pool_layer_zero = True
        self.assertAlmostEqual(float(model.aggregate()[0]), 7.0 / 3.0, places=5)
        model.pool_layer_zero = False
        self.assertAlmostEqual(float(model.aggregate()[0]), 3.0, places=5)

    def test_external_tables_cannot_be_trained_here(self):
        from models.ExternalEmbeddings import Trainer

        dataset = SimpleNamespace(num_users=1, num_items=1)
        trainer = Trainer(None, {"embedding_size": 2}, dataset, torch.device("cpu"), None)
        with self.assertRaises(RuntimeError):
            trainer.train()


@unittest.skipUnless((ROOT / "log" / "p0" / "taxprocl").is_dir(), "needs the local run manifests")
class ComparatorPlanTests(unittest.TestCase):
    def test_each_method_has_twelve_cells_of_three_seeds(self):
        from tools.experiments.run_comparators import PACKAGES, plan

        for package in PACKAGES:
            jobs = plan(package)
            self.assertEqual(len({j.cell for j in jobs}), 12)
            self.assertEqual({j.seed for j in jobs}, {42, 0, 1})
            self.assertEqual(len({j.run_id for j in jobs}), 36)

    def test_run_directories_follow_the_log_layout(self):
        from tools.experiments.run_comparators import plan

        paths = {j.run_dir.as_posix() for j in plan("comparator-galore")}
        self.assertIn("log/p0/extended/office-products/GALORE/grid-edges3-drop0.3/seed42", paths)
        paths = {j.run_dir.as_posix() for j in plan("comparator-kgcl")}
        self.assertIn("log/p0/extended/office-products/KGCL/grid-temp0.2-ssl0.1/seed0", paths)

    def test_the_grids_hold_the_author_settings(self):
        from tools.experiments.run_comparators import plan

        self.assertTrue(any(j.cell == "grid-temp0.2-ssl0.1" for j in plan("comparator-kgcl")))
        galore = {dict(j.config)["edge_num"] for j in plan("comparator-galore")},             {dict(j.config)["drop_rate"] for j in plan("comparator-galore")}
        self.assertEqual(galore, ({"1", "3", "5", "10"}, {"0.1", "0.3", "0.5"}))

    def test_category_triples_follow_the_path_and_skip_markup(self):
        from tools.experiments.run_comparators import category_triples

        rows = [{"item_id": 0, "final_leaf": "A > B > C"}, {"item_id": 1, "final_leaf": "A > B"},
                {"item_id": 2, "final_leaf": None}, {"item_id": 3, "final_leaf": "<img src=x> > A > B"}]
        triples = category_triples(rows)
        self.assertEqual([t for t in triples if t[0] == 0], [(0, 0, 0), (0, 1, 1), (0, 2, 2)])
        self.assertEqual([t for t in triples if t[0] == 1], [(1, 0, 1), (1, 1, 2)])
        self.assertEqual([t for t in triples if t[0] == 2], [])
        self.assertEqual([t for t in triples if t[0] == 3], [(3, 0, 1), (3, 1, 2)])

    def test_tables_are_padded_with_zero_vectors(self):
        from tools.experiments.run_comparators import pad

        table = pad(np.ones((2, 3), dtype=np.float32), 4)
        self.assertEqual(table.shape, (4, 3))
        self.assertEqual(float(table[2:].abs().sum()), 0.0)

    def test_test_split_stays_closed_without_a_committed_selection(self):
        from tools.experiments import run_comparators

        with mock.patch.object(run_comparators, "selection_file", return_value=Path("results/extended/_absent.json")):
            for name in ("kgcl", "galore"):
                with self.assertRaises(SystemExit):
                    run_comparators.open_test(name, "cpu")

    @unittest.skipUnless((ROOT / "log" / "p0" / "confirmatory").is_dir(), "needs the local run manifests")
    def test_galore_opens_only_with_its_starting_point_complete(self):
        from tools.experiments import run_comparators

        selection = {"GALORE": {"rule_R": "edges2-drop0.1"}}
        with mock.patch.object(run_comparators, "committed_selection", return_value=selection), \
                mock.patch.object(run_comparators, "run_status", return_value="running"), \
                mock.patch.object(run_comparators.subprocess, "run") as evaluate:
            with self.assertRaises(SystemExit):
                run_comparators.open_test("galore", "cpu")
            evaluate.assert_not_called()

    def test_each_method_has_its_own_selection_file(self):
        from tools.experiments.run_comparators import METHODS, PACKAGES, selection_file

        self.assertEqual({selection_file(PACKAGES[package]).name for package in METHODS.values()},
                         {"selection_kgcl.json", "selection_galore.json"})


class AnalysisTests(unittest.TestCase):
    def test_six_run_classification(self):
        self.assertEqual(classify([[1, 1, 1, 1, 1, -0.1]]), "stable positive")
        self.assertEqual(classify([[1, 1, 1, 1, -0.1, -0.1]]), "unstable")
        self.assertEqual(classify([[-1, -1, -1, -1, -1, 0.1]]), "stable negative")

    def test_nine_run_classification_needs_every_split_mean(self):
        self.assertEqual(classify([[1, 1, 1], [1, 1, 1], [1, 1, -0.1]]), "stable positive")
        self.assertEqual(classify([[1, 1, 1], [1, 1, 1], [-3, 1, 1]]), "unstable")
        self.assertEqual(classify([[1, 1, 1], [1, 1, -0.1], [1, -0.1, 1]]), "unstable")

    def test_reduced_designs_use_the_stated_counts(self):
        self.assertEqual(classify([[1, 1, 1], [1, 1, -0.1]]), "stable positive")
        self.assertEqual(classify([[1, 1, 1, 1, -0.1]]), "stable positive")

    def test_sign_test_reproduces_the_protocol_values(self):
        self.assertAlmostEqual(sign_test([1] * 6), 0.03125)
        self.assertAlmostEqual(sign_test([1] * 5 + [-1]), 0.21875)
        self.assertAlmostEqual(sign_test([1] * 9), 2 / 512)
        self.assertAlmostEqual(sign_test([1] * 8 + [-1]), 20 / 512)
        self.assertAlmostEqual(sign_test([1] * 7 + [-1] * 2), 92 / 512)

    def test_wording_scale(self):
        self.assertEqual(wording({"holm_significant": True, "excludes_zero": True}), "evidence")
        self.assertEqual(wording({"holm_significant": False, "excludes_zero": True}), "nominal signal")
        self.assertEqual(wording({"excludes_zero": False}), "not detected")

    def test_upper_edge_of_the_grid(self):
        edge = at_upper_edge({"complete_configuration": "temp0.2-tempuser0.3", "simgcl": "temp0.3-eps0.1"})
        self.assertEqual(edge, {"complete_configuration": False, "simgcl": False})
        edge = at_upper_edge({"complete_configuration": "temp0.15-tempuser0.4", "simgcl": "temp0.4-eps0.05"})
        self.assertEqual(edge, {"complete_configuration": True, "simgcl": True})
        self.assertTrue(at_upper_edge({"complete_configuration": "temp0.25-tempuser0.3",
                                       "simgcl": "temp0.2-eps0.1"})["complete_configuration"])

    def test_mean_std_uses_the_sample_deviation(self):
        self.assertAlmostEqual(mean_std([1.0, 2.0, 3.0])["std"], 1.0)

    def test_starting_point_wording(self):
        self.assertEqual(starting_point_wording({"mean_diff": -0.001, "excludes_zero": True}),
                         "did not improve over its starting point")
        self.assertEqual(starting_point_wording({"mean_diff": 0.0, "excludes_zero": False}),
                         "did not improve over its starting point")
        self.assertEqual(starting_point_wording({"mean_diff": 0.001, "excludes_zero": False}),
                         "no detectable improvement")
        self.assertEqual(starting_point_wording({"mean_diff": 0.001, "excludes_zero": True}),
                         "improved over its starting point")

    def test_rank_measure_table_covers_every_pair_group_and_dataset(self):
        groups = ("near_cold", "long_tail", "warm", "overall")
        def measures(value):
            return {group: {"recall_50": value, "recall_100": value, "ndcg": value, "percentile": value}
                    for group in groups}
        arms = {"V0": 0.1, "V1": 0.2, "V2": 0.4, "V3": 0.8}
        runs = {arm: {seed: {"validation_measures": measures(value + int(seed) / 1000)}
                      for seed in ("42", "0", "1")} for arm, value in arms.items()}
        ratio = {"delta_arm": dict.fromkeys(groups, 0.5), "delta_seed": dict.fromkeys(groups, 0.25),
                 "r": dict.fromkeys(groups, 2.0)}
        block = {"runs": runs, "ranking_change_ratio": {pair: ratio for pair in ("V2-V0", "V3-V1", "V1-V0", "V3-V2")}}
        table = rank_measure_table({"datasets": {"musical-instruments": block}})
        self.assertEqual(len(table["rows"]), 4 * 4)
        self.assertEqual(len(table["not_measured"]), 4)
        row = next(r for r in table["rows"] if r["arms"] == "V3-V2" and r["group"] == "long_tail")
        self.assertEqual(row["role"], "primary")
        self.assertAlmostEqual(row["recall_50"]["mean"], 0.4)
        self.assertAlmostEqual(row["ndcg_100"]["per_seed"]["0"], 0.4)
        self.assertEqual(row["r"], 2.0)


class EnvironmentLabelTests(unittest.TestCase):
    def label(self, gpu, platform, log_line):
        from tools.experiments.run_extended import environment_label

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            if log_line is not None:
                (run_dir / "training.log").write_text("2026 | INFO | main | Start Training\n"
                                                      f"2026 | INFO | main | Training log: {log_line}\n",
                                                      encoding="utf-8")
            return environment_label(run_dir, {"gpu_name": gpu, "platform": platform})

    def test_gpu_platform_and_log_paths_must_agree(self):
        windows, linux = "Windows-10-10.0.19045-SP0", "Linux-5.15.0-187-generic-x86_64-with-glibc2.35"
        self.assertEqual(self.label("NVIDIA GeForce RTX 4060 Ti", windows, r"log\p0\run\training.log"), "A")
        self.assertEqual(self.label("NVIDIA GeForce RTX 3090 Ti", linux, "log/p0/run/training.log"), "B")
        self.assertEqual(self.label("NVIDIA GeForce RTX 4060 Ti", windows, "log/p0/run/training.log"), "unknown")
        self.assertEqual(self.label("NVIDIA GeForce RTX 4060 Ti", linux, None), "unknown")
        self.assertEqual(self.label("NVIDIA GeForce RTX 4060 Ti", windows, None), "A")


class DiagnosticHelperTests(unittest.TestCase):
    def test_jaccard_of_top_lists(self):
        left = np.arange(20).reshape(1, 20)
        right = np.arange(10, 30).reshape(1, 20)
        self.assertAlmostEqual(float(jaccard(left, right)[0]), 10 / 30)
        self.assertAlmostEqual(float(jaccard(left, left)[0]), 1.0)

    def test_degree_masks_partition_observed_items(self):
        masks = degree_masks([0, 1, 5, 6, 10, 11])
        self.assertEqual(masks["near_cold"].tolist(), [False, True, True, False, False, False])
        self.assertEqual(masks["mid_tail"].tolist(), [False, False, False, True, True, False])
        self.assertEqual(masks["warm"].tolist(), [False, False, False, False, False, True])

    def test_group_and_layer_means(self):
        masks = degree_masks([1, 7, 20])
        values = np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        result = by_bin_and_layer(values, masks)
        self.assertEqual(result["near_cold"], {"per_layer": [1.0, 3.0], "mean": 2.0})

    def test_embedding_change_measures(self):
        masks = degree_masks([1])
        reference = np.asarray([[3.0, 4.0]])
        self.assertAlmostEqual(embedding_shift(reference, 2.0 * reference, masks)["near_cold"], 1.0)
        self.assertAlmostEqual(embedding_cosine(reference, 2.0 * reference, masks)["near_cold"], 1.0)

    def scaling_runs(self, cold_ratio):
        def norm(values):
            return {"near_cold": {"per_layer": values}, "mid_tail": {"per_layer": [1.0]}, "warm": {"per_layer": [1.0]}}

        fixed = {"42": {"realized_perturbation_norm": norm([1.0])}}
        adaptive = {"42": {"realized_perturbation_norm": norm([cold_ratio])}}
        return {"V0": fixed, "V1": fixed, "V2": adaptive, "V3": adaptive}

    def test_scaling_check_passes_within_two_percent(self):
        rows = scaling_check(self.scaling_runs(1.52), 1.5, (42,))
        self.assertTrue(all(row["within_tolerance"] for row in rows))

    def test_scaling_check_fails_outside_two_percent(self):
        rows = scaling_check(self.scaling_runs(1.6), 1.5, (42,))
        failed = {(row["pair"], row["group"]) for row in rows if not row["within_tolerance"]}
        self.assertEqual(failed, {("V2/V0", "near_cold"), ("V3/V1", "near_cold")})

    def test_probe_measures_the_realized_perturbation_per_layer(self):
        with patched_improved_dependencies():
            model = TaxProCLImproved(improved_config(), ToyDataset(), torch.device("cpu"))
        model.config = {**model.config, "GCN_layer": 2}
        embedding = torch.cat([torch.zeros(3, 4), torch.ones(4, 4)])
        added = torch.cat([torch.zeros(3, 4), torch.full((4, 4), 0.5)])
        model._perturb_items = lambda value, layers, view_id=None: value + added
        with PerturbationProbe(model) as probe:
            for _ in range(4):
                model._perturb_items(embedding, 2, view_id=0)
        norm = probe.norm / 2
        torch.testing.assert_close(norm, torch.ones_like(norm, dtype=torch.float64))
        self.assertAlmostEqual(float((probe.ratio / 2)[0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
