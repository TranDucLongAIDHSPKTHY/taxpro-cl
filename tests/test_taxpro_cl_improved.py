import importlib
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

from tests.helpers import (
    ToyDataset,
    base_config,
    patched_taxpro_dependencies,
    patched_taxpro_dependencies_mixture,
)
from utility.utility_function.taxonomy_contrastive import (
    degree_epsilon_scale,
    item_level_info_nce,
    leaf_aware_info_nce,
)

# "TaxPro-CL" is not a valid Python identifier, so the module can only be
# reached via importlib, exactly like main.py's dynamic model loader does.
_improved_module = importlib.import_module("models.TaxPro-CL")
TaxProCLImproved = _improved_module.TaxProCLImproved


def improved_config(**overrides):
    config = base_config()
    config.update({
        "use_adaptive_epsilon": "true",
        "use_leaf_aware_infonce": "true",
        "gamma_cold": "1.5",
        "gamma_warm": "0.5",
        "same_leaf_weight": "0.05",
        "degree_near_cold_max": "5",
        "degree_long_tail_max": "10",
    })
    config.update(overrides)
    return config


_EVALUATION_MANIFEST = {
    "source_hashes": {"train": "train", "validation": "validation", "test": "test"}
}


@contextmanager
def patched_improved_dependencies():
    degree_payload = {"degrees": [2, 6, 12, 1]}  # aligned with ToyDataset's 4 items
    # models/TaxPro-CL.py is self-contained (see its class docstring), so its
    # own read_json name now serves BOTH the base construction's evaluation
    # manifest read AND this subclass's item_degrees_train.json read, in
    # that call order -- side_effect list mirrors that order.
    with patched_taxpro_dependencies() as fake, mock.patch(
        "models.TaxPro-CL.evaluation_protocol_dir"
    ) as directory, mock.patch(
        "models.TaxPro-CL.read_json",
        side_effect=[_EVALUATION_MANIFEST, degree_payload],
    ):
        directory.return_value.__truediv__.return_value = "item_degrees_train.json"
        yield fake


@contextmanager
def patched_improved_dependencies_mixture():
    degree_payload = {"degrees": [2, 6, 12, 1]}
    with patched_taxpro_dependencies_mixture() as (leaf, parent), mock.patch(
        "models.TaxPro-CL.evaluation_protocol_dir"
    ) as directory, mock.patch(
        "models.TaxPro-CL.read_json",
        side_effect=[_EVALUATION_MANIFEST, degree_payload],
    ):
        directory.return_value.__truediv__.return_value = "item_degrees_train.json"
        yield leaf, parent


class DegreeAdaptiveViewsTests(unittest.TestCase):
    def test_degree_epsilon_scale_buckets_match_group_evaluator_thresholds(self):
        degree = torch.tensor([1, 5, 6, 10, 11, 50])
        scale = degree_epsilon_scale(
            degree, gamma_cold=1.5, gamma_warm=0.5, near_cold_max=5, long_tail_max=10
        )
        self.assertTrue(
            torch.allclose(scale, torch.tensor([1.5, 1.5, 1.0, 1.0, 0.5, 0.5]))
        )

    def test_zero_same_leaf_weight_reduces_exactly_to_item_level_info_nce(self):
        embeddings = torch.randn(6, 4)
        leaves = torch.tensor([0, 0, 1, 1, 2, 0])
        gen = torch.Generator().manual_seed(1)
        view_a = embeddings + 0.05 * torch.randn(6, 4, generator=gen)
        view_b = embeddings + 0.05 * torch.randn(6, 4, generator=gen)
        loss_original = item_level_info_nce(view_a, view_b, temperature=0.2)
        loss_leaf_aware = leaf_aware_info_nce(
            view_a, view_b, leaves, temperature=0.2, same_leaf_weight=0.0
        )
        self.assertAlmostEqual(float(loss_original), float(loss_leaf_aware), places=5)


class TaxProCLImprovedModelTests(unittest.TestCase):
    def test_aggregate_excludes_layer0_like_simgcl(self):
        """SimGCL backbone convention: layer-0 (ego embedding) is excluded
        from the final mean-pool, unlike vanilla LightGCN/TaxProCL."""
        with patched_improved_dependencies():
            model = TaxProCLImproved(
                improved_config(), ToyDataset(), torch.device("cpu")
            )
        gcn_layers = int(model.config["GCN_layer"])
        all_embedding = torch.cat(
            [model.user_embedding.weight, model.item_embedding.weight]
        )
        manual_layers = []
        current = all_embedding
        for _ in range(gcn_layers):
            current = torch.sparse.mm(model.Graph, current)
            manual_layers.append(current)
        expected = torch.stack(manual_layers, dim=1).mean(dim=1)
        users, items = model.aggregate(perturbed=False)
        actual = torch.cat([users, items])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_perturbed_aggregate_is_unperturbed_before_prototype_init(self):
        """Prototype isn't ready during warm-start, so aggregate(True) must
        fall back to the clean pass -- no NaNs from an uninitialized bank."""
        with patched_improved_dependencies():
            model = TaxProCLImproved(
                improved_config(), ToyDataset(), torch.device("cpu")
            )
        clean_users, clean_items = model.aggregate(perturbed=False)
        pert_users, pert_items = model.aggregate(perturbed=True)
        self.assertTrue(torch.equal(clean_users, pert_users))
        self.assertTrue(torch.equal(clean_items, pert_items))

    def test_two_perturbed_views_differ_after_prototype_init(self):
        with patched_improved_dependencies():
            model = TaxProCLImproved(
                improved_config(), ToyDataset(), torch.device("cpu")
            )
        model.initialize_prototypes(epoch=2)
        _, items_a = model.aggregate(perturbed=True)
        _, items_b = model.aggregate(perturbed=True)
        self.assertFalse(torch.equal(items_a, items_b))
        # Sanity bound only (not a tight derivation): with gamma_cold up to
        # 1.5x and GCN_layer steps summing via the triangle inequality, the
        # per-layer division keeps cumulative displacement on the same
        # order as epsilon_max rather than epsilon_max * GCN_layer.
        _, clean_items = model.aggregate(perturbed=False)
        displacement = (items_a - clean_items).norm(dim=1)
        self.assertTrue(torch.isfinite(displacement).all())
        self.assertTrue(bool((displacement <= 2.0 * model.epsilon_max).all()))

    def test_warm_start_then_joint_phase_stays_finite_with_full_al_taxcl(self):
        with patched_improved_dependencies():
            model = TaxProCLImproved(
                improved_config(), ToyDataset(), torch.device("cpu")
            )
        users = torch.tensor([0, 1])
        positive = torch.tensor([0, 2])
        negative = torch.tensor([3, 0])
        self.assertEqual(float(model(users, positive, negative, epoch=1)[2]), 0.0)
        self.assertFalse(bool(model.prototype_bank.initialized.item()))
        model.initialize_prototypes(epoch=2)

        losses = model(
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 0, 2]),
            torch.tensor([3, 2, 0]),
            epoch=3,
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses))
        self.assertGreater(model.last_batch_statistics["cl_item_count"], 0)
        self.assertTrue(model.last_batch_statistics["use_adaptive_epsilon"])
        self.assertTrue(model.last_batch_statistics["use_leaf_aware_infonce"])

        rating = model.get_rating_for_test(users)
        self.assertEqual(tuple(rating.shape), (2, 4))
        self.assertTrue(torch.isfinite(rating).all())

    def test_checkpoint_metadata_round_trip(self):
        with patched_improved_dependencies():
            model = TaxProCLImproved(
                improved_config(), ToyDataset(), torch.device("cpu")
            )
        metadata = model.checkpoint_metadata()
        self.assertEqual(metadata["model"], "TaxPro-CL")
        model.validate_checkpoint_metadata(metadata)
        broken = dict(metadata, gamma_cold=999.0)
        with self.assertRaises(ValueError):
            model.validate_checkpoint_metadata(broken)

    def test_rejects_non_leaf_prototype_mode(self):
        with self.assertRaises(ValueError):
            with patched_improved_dependencies():
                TaxProCLImproved(
                    improved_config(prototype_mode="parent"),
                    ToyDataset(),
                    torch.device("cpu"),
                )


class TaxProCLImprovedPeerDirectionTests(unittest.TestCase):
    """direction_source="peer": stochastic same-leaf peer direction, an
    alternative to the fixed EMA prototype target -- addresses the "both
    views point the same way" diversity ceiling while staying single-level
    leaf-taxonomy (no parent/multi-level signal, no new loss term, no
    gating)."""

    def test_build_leaf_peer_index_groups_and_ranks_correctly(self):
        # ToyDataset leaves: [0, 0, 1, -1(invalid)] -- items 0,1 share leaf 0.
        leaf_ids = torch.tensor([0, 0, 1, -1])
        valid = torch.tensor([True, True, True, False])
        sort_idx, group_start, group_size, rank = _improved_module._build_leaf_peer_index(
            leaf_ids, valid
        )
        # Items 0 and 1 (leaf 0) form a 2-member group; item 2 (leaf 1) and
        # item 3 (invalid) are both singleton groups.
        self.assertEqual(int(group_size[0]), 2)
        self.assertEqual(int(group_size[1]), 2)
        self.assertEqual(int(group_size[2]), 1)
        self.assertEqual(int(group_size[3]), 1)
        # Items 0 and 1 must resolve to each other via the group structure.
        pos0 = int(group_start[0]) + int(rank[0])
        pos1 = int(group_start[1]) + int(rank[1])
        self.assertEqual(int(sort_idx[pos0]), 0)
        self.assertEqual(int(sort_idx[pos1]), 1)

    def test_peer_direction_only_active_items_are_same_leaf_pairs(self):
        # Amazon-book-style config so mixture guard doesn't interfere; here
        # we only exercise direction_source=peer with prototype_mode=leaf.
        with patched_improved_dependencies():
            model = TaxProCLImproved(
                improved_config(direction_source="peer"),
                ToyDataset(),
                torch.device("cpu"),
            )
        model.initialize_prototypes(epoch=2)
        all_embedding = torch.cat(
            [model.user_embedding.weight, model.item_embedding.weight]
        )
        torch.manual_seed(7)
        perturbed = model._perturb_items(all_embedding.clone(), gcn_layers=1)
        _, clean_items = torch.split(
            all_embedding, [model.dataset.num_users, model.dataset.num_items]
        )
        _, perturbed_items = torch.split(
            perturbed, [model.dataset.num_users, model.dataset.num_items]
        )
        displacement = (perturbed_items - clean_items).norm(dim=1)
        # Item 2's leaf (leaf 1) has only itself as a valid member -> no
        # peer exists -> must be excluded (zero displacement).
        self.assertAlmostEqual(float(displacement[2]), 0.0, places=6)
        # Item 3 is taxonomy-invalid -> also excluded.
        self.assertAlmostEqual(float(displacement[3]), 0.0, places=6)
        # Items 0 and 1 (leaf 0, 2 valid members) must have a real peer and
        # therefore nonzero displacement.
        self.assertGreater(float(displacement[0]), 0.0)
        self.assertGreater(float(displacement[1]), 0.0)

    def test_peer_direction_differs_across_independent_calls(self):
        with patched_improved_dependencies():
            model = TaxProCLImproved(
                improved_config(direction_source="peer"),
                ToyDataset(),
                torch.device("cpu"),
            )
        model.initialize_prototypes(epoch=2)
        _, items_a = model.aggregate(perturbed=True)
        _, items_b = model.aggregate(perturbed=True)
        # With only a 2-member leaf group available in ToyDataset, the peer
        # choice itself can't vary (item0's only peer is item1) -- but the
        # independently-sampled epsilon per call must still make the two
        # views differ, exactly like the prototype-direction path already
        # guarantees (see test_two_perturbed_views_differ_after_prototype_init).
        self.assertFalse(torch.equal(items_a, items_b))

    def test_rejects_invalid_direction_source(self):
        with self.assertRaises(ValueError):
            with patched_improved_dependencies():
                TaxProCLImproved(
                    improved_config(direction_source="bogus"),
                    ToyDataset(),
                    torch.device("cpu"),
                )

    def test_peer_direction_combined_with_mixture_prototype_mode(self):
        # v7 combines direction_source=peer (leaf-level target) with
        # prototype_mode=mixture (blends in the parent prototype direction)
        # -- untested combination before this, so verify it runs end-to-end
        # without shape/device errors and still produces two differing,
        # finite views.
        with patched_improved_dependencies_mixture():
            model = TaxProCLImproved(
                improved_config(
                    direction_source="peer",
                    prototype_mode="mixture",
                    dataset="amazon-book",
                    ssl_lambda="2.0",
                ),
                ToyDataset(),
                torch.device("cpu"),
            )
        model.initialize_prototypes(epoch=2)
        losses = model(
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 0, 2]),
            torch.tensor([3, 2, 0]),
            epoch=3,
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses))
        _, items_a = model.aggregate(perturbed=True)
        _, items_b = model.aggregate(perturbed=True)
        self.assertTrue(torch.isfinite(items_a).all())
        self.assertFalse(torch.equal(items_a, items_b))


class TaxProCLImprovedMixtureModeTests(unittest.TestCase):
    """prototype_mode="mixture": fixed convex mixture of leaf and parent
    prototype directions. Still 100% taxonomy-derived, reuses the locked
    baseline's mixture infrastructure."""

    def test_mixture_requires_amazon_book_dataset(self):
        with self.assertRaises(ValueError):
            with patched_improved_dependencies_mixture():
                TaxProCLImproved(
                    improved_config(prototype_mode="mixture"),
                    ToyDataset(),
                    torch.device("cpu"),
                )

    def test_mixture_direction_matches_alpha_blend_formula(self):
        with patched_improved_dependencies_mixture():
            model = TaxProCLImproved(
                improved_config(
                    prototype_mode="mixture",
                    dataset="amazon-book",
                    mixture_alpha="0.5",
                    use_adaptive_epsilon="false",
                ),
                ToyDataset(),
                torch.device("cpu"),
            )
        model.initialize_prototypes(epoch=2)
        self.assertTrue(bool(model.parent_prototype_bank.initialized.item()))
        # Parent bucket [0, 0, 0, -1] differs from leaf split [0, 0, 1, -1],
        # so the two prototype banks must disagree for this to be a real test.
        self.assertFalse(
            torch.equal(
                model.prototype_bank.prototypes, model.parent_prototype_bank.prototypes
            )
        )

        all_embedding = torch.cat(
            [model.user_embedding.weight, model.item_embedding.weight]
        )
        gcn_layers = int(model.config["GCN_layer"])

        torch.manual_seed(123)
        actual = model._perturb_items(all_embedding.clone(), gcn_layers)

        _, items = torch.split(
            all_embedding, [model.dataset.num_users, model.dataset.num_items]
        )
        safe_leaf = model.item_to_leaf_id.clamp_min(0).long()
        safe_parent = model.item_to_parent_id.clamp_min(0).long()
        leaf_direction = _improved_module._normalized_direction(
            model.prototype_bank.prototypes[safe_leaf] - items, model.delta
        )
        parent_direction = _improved_module._normalized_direction(
            model.parent_prototype_bank.prototypes[safe_parent] - items, model.delta
        )
        expected_direction = (
            model.mixture_alpha * leaf_direction
            + (1.0 - model.mixture_alpha) * parent_direction
        )
        valid_mask = model.valid_train_mask & model.valid_parent_train_mask
        expected_direction = expected_direction * valid_mask.unsqueeze(1).to(
            expected_direction.dtype
        )

        torch.manual_seed(123)
        per_layer_epsilon_max = model.epsilon_max / float(gcn_layers)
        epsilon = torch.rand((items.shape[0], 1), dtype=items.dtype) * per_layer_epsilon_max
        expected_items = items + epsilon * expected_direction

        _, actual_items = torch.split(
            actual, [model.dataset.num_users, model.dataset.num_items]
        )
        self.assertTrue(torch.allclose(actual_items, expected_items, atol=1e-6))

    def test_forward_updates_parent_prototype_bank_in_mixture_mode(self):
        with patched_improved_dependencies_mixture():
            model = TaxProCLImproved(
                improved_config(prototype_mode="mixture", dataset="amazon-book"),
                ToyDataset(),
                torch.device("cpu"),
            )
        model.initialize_prototypes(epoch=2)
        model(
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 0, 2]),
            torch.tensor([3, 2, 0]),
            epoch=3,
        )
        self.assertIn("parent_leaf_update_count", model.last_batch_statistics)
        self.assertGreater(model.last_batch_statistics["parent_leaf_update_count"], 0)


if __name__ == "__main__":
    unittest.main()
