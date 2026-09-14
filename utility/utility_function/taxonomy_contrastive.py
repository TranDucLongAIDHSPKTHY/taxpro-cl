"""Prototype memory, taxonomy-guided views and item-level InfoNCE for TaxPro-CL."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeBank(nn.Module):
    """A non-learnable, checkpointable leaf-level prototype memory."""

    def __init__(self, num_prototypes, embedding_size, mu=0.9):
        super().__init__()
        if int(num_prototypes) <= 0:
            raise ValueError("num_prototypes must be positive")
        if not 0.0 <= float(mu) < 1.0:
            raise ValueError("mu must be in [0, 1)")
        self.mu = float(mu)
        self.register_buffer(
            "prototypes",
            torch.zeros(int(num_prototypes), int(embedding_size)),
        )
        self.register_buffer(
            "support", torch.zeros(int(num_prototypes), dtype=torch.long)
        )
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("init_epoch", torch.tensor(-1, dtype=torch.long))
        # Only used by ema_update_snapshot (prototype_weighting=leaf_uniform,
        # 2026-09-09 addition, GVHD A3 "unique-item EMA" variant): tracks which
        # epoch the snapshot was last taken so the (expensive, full-catalog)
        # recompute happens once per epoch, not once per batch. persistent=False
        # so it is NOT part of state_dict -- otherwise every checkpoint saved
        # before this addition fails to load_state_dict(strict=True) with a
        # missing-key error. Losing this on resume just costs one redundant
        # snapshot recompute on the first post-resume batch; harmless.
        self.register_buffer(
            "last_snapshot_epoch", torch.tensor(-1, dtype=torch.long), persistent=False
        )

    @torch.no_grad()
    def initialize(self, item_embeddings, item_to_leaf_id, valid_train_mask, epoch):
        if bool(self.initialized.item()):
            raise RuntimeError("Prototype bank is already initialized")
        if item_embeddings.ndim != 2:
            raise ValueError("item_embeddings must be a matrix")
        mask = valid_train_mask.bool() & (item_to_leaf_id >= 0)
        item_ids = torch.nonzero(mask, as_tuple=False).flatten()
        if item_ids.numel() == 0:
            raise ValueError("No valid train-observed taxonomy items")
        leaves = item_to_leaf_id[item_ids].long()
        if int(leaves.max()) >= self.prototypes.shape[0]:
            raise ValueError("Leaf id exceeds prototype bank")
        sums = torch.zeros_like(self.prototypes)
        counts = torch.zeros_like(self.support)
        sums.index_add_(0, leaves, item_embeddings[item_ids].detach())
        counts.index_add_(
            0,
            leaves,
            torch.ones_like(leaves, dtype=counts.dtype),
        )
        active = counts > 0
        self.prototypes.zero_()
        self.prototypes[active] = sums[active] / counts[active].unsqueeze(1)
        self.support.copy_(counts)
        self.initialized.fill_(True)
        self.init_epoch.fill_(int(epoch))
        norms = self.prototypes[active].norm(dim=1)
        return {
            "init_epoch": int(epoch),
            "number_of_prototypes": int(active.sum().item()),
            "item_support": int(counts.sum().item()),
            "prototype_norm_min": float(norms.min().item()),
            "prototype_norm_mean": float(norms.mean().item()),
            "prototype_norm_max": float(norms.max().item()),
        }

    @torch.no_grad()
    def ema_update(
        self,
        item_embeddings,
        positive_item_ids,
        item_to_leaf_id,
        valid_train_mask,
    ):
        if not bool(self.initialized.item()):
            raise RuntimeError("Prototype bank must be initialized before EMA")
        unique_items = torch.unique(positive_item_ids.long())
        valid = (
            valid_train_mask[unique_items].bool()
            & (item_to_leaf_id[unique_items] >= 0)
        )
        valid_items = unique_items[valid]
        invalid_count = int(unique_items.numel() - valid_items.numel())
        if valid_items.numel() == 0:
            return {
                "leaf_update_count": 0,
                "item_update_count": 0,
                "prototype_norm_mean": 0.0,
                "prototype_drift_mean": 0.0,
                "inactive_leaf_count": int(self.prototypes.shape[0]),
                "invalid_taxonomy_item_count": invalid_count,
            }
        leaves = item_to_leaf_id[valid_items].long()
        unique_leaves, inverse = torch.unique(leaves, sorted=True, return_inverse=True)
        sums = torch.zeros(
            unique_leaves.numel(),
            self.prototypes.shape[1],
            dtype=self.prototypes.dtype,
            device=self.prototypes.device,
        )
        counts = torch.zeros(
            unique_leaves.numel(),
            dtype=self.prototypes.dtype,
            device=self.prototypes.device,
        )
        detached = item_embeddings[valid_items].detach()
        sums.index_add_(0, inverse, detached)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=counts.dtype))
        means = sums / counts.unsqueeze(1)
        old = self.prototypes[unique_leaves].clone()
        updated = self.mu * old + (1.0 - self.mu) * means
        self.prototypes[unique_leaves] = updated
        drift = (updated - old).norm(dim=1)
        norms = updated.norm(dim=1)
        return {
            "leaf_update_count": int(unique_leaves.numel()),
            "item_update_count": int(valid_items.numel()),
            "prototype_norm_mean": float(norms.mean().item()),
            "prototype_drift_mean": float(drift.mean().item()),
            "inactive_leaf_count": int(
                self.prototypes.shape[0] - unique_leaves.numel()
            ),
            "invalid_taxonomy_item_count": invalid_count,
        }

    @torch.no_grad()
    def ema_update_snapshot(
        self,
        all_item_embeddings,
        item_to_leaf_id,
        valid_train_mask,
        epoch,
    ):
        """prototype_weighting=leaf_uniform (2026-09-09, GVHD A3 "unique-item
        EMA" variant): unlike ema_update above (which pulls p_l toward
        whichever items happened to be positive in *this* batch -- an item
        that appears as positive in more batches over an epoch gets more
        pulls, i.e. an interaction-frequency-weighted centroid, exactly the
        A3 review concern), this recomputes p_l's EMA target as the
        UNWEIGHTED mean over every valid item currently in that leaf,
        regardless of whether it was a positive this batch. Gated to run at
        most once per epoch (self.last_snapshot_epoch) since it touches the
        full catalog, not just batch positives -- doing this every batch
        would be wasteful and the embeddings barely move batch-to-batch."""
        if not bool(self.initialized.item()):
            raise RuntimeError("Prototype bank must be initialized before EMA")
        if int(epoch) == int(self.last_snapshot_epoch.item()):
            return {
                "leaf_update_count": 0,
                "item_update_count": 0,
                "prototype_norm_mean": float(self.prototypes.norm(dim=1).mean().item()),
                "prototype_drift_mean": 0.0,
                "inactive_leaf_count": 0,
                "invalid_taxonomy_item_count": 0,
            }
        mask = valid_train_mask.bool() & (item_to_leaf_id >= 0)
        item_ids = torch.nonzero(mask, as_tuple=False).flatten()
        invalid_count = int(mask.numel() - mask.sum().item())
        if item_ids.numel() == 0:
            return {
                "leaf_update_count": 0,
                "item_update_count": 0,
                "prototype_norm_mean": 0.0,
                "prototype_drift_mean": 0.0,
                "inactive_leaf_count": int(self.prototypes.shape[0]),
                "invalid_taxonomy_item_count": invalid_count,
            }
        leaves = item_to_leaf_id[item_ids].long()
        unique_leaves, inverse = torch.unique(leaves, sorted=True, return_inverse=True)
        sums = torch.zeros(
            unique_leaves.numel(),
            self.prototypes.shape[1],
            dtype=self.prototypes.dtype,
            device=self.prototypes.device,
        )
        counts = torch.zeros(
            unique_leaves.numel(),
            dtype=self.prototypes.dtype,
            device=self.prototypes.device,
        )
        detached = all_item_embeddings[item_ids].detach()
        sums.index_add_(0, inverse, detached)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=counts.dtype))
        means = sums / counts.unsqueeze(1)
        old = self.prototypes[unique_leaves].clone()
        updated = self.mu * old + (1.0 - self.mu) * means
        self.prototypes[unique_leaves] = updated
        self.last_snapshot_epoch.fill_(int(epoch))
        drift = (updated - old).norm(dim=1)
        norms = updated.norm(dim=1)
        return {
            "leaf_update_count": int(unique_leaves.numel()),
            "item_update_count": int(item_ids.numel()),
            "prototype_norm_mean": float(norms.mean().item()),
            "prototype_drift_mean": float(drift.mean().item()),
            "inactive_leaf_count": int(
                self.prototypes.shape[0] - unique_leaves.numel()
            ),
            "invalid_taxonomy_item_count": invalid_count,
        }


def _normalized_direction(vector, delta):
    return vector / (vector.norm(dim=1, keepdim=True) + float(delta))


def create_item_views(
    item_embeddings,
    leaf_ids,
    valid_mask,
    prototypes,
    epsilon_max=0.1,
    delta=1e-8,
    direction_mode="taxonomy",
    generator=None,
):
    """Create independent item-level views; invalid rows remain unperturbed."""
    if float(epsilon_max) < 0.0:
        raise ValueError("epsilon_max must be non-negative")
    if float(delta) <= 0.0:
        raise ValueError("delta must be positive")
    if direction_mode not in {"taxonomy", "random"}:
        raise ValueError("direction_mode must be taxonomy or random")
    valid_mask = valid_mask.bool() & (leaf_ids >= 0)
    epsilon_a = torch.rand(
        (item_embeddings.shape[0], 1),
        dtype=item_embeddings.dtype,
        device=item_embeddings.device,
        generator=generator,
    ) * float(epsilon_max)
    epsilon_b = torch.rand(
        (item_embeddings.shape[0], 1),
        dtype=item_embeddings.dtype,
        device=item_embeddings.device,
        generator=generator,
    ) * float(epsilon_max)
    if direction_mode == "taxonomy":
        safe_leaf_ids = leaf_ids.clamp_min(0).long()
        raw_direction = prototypes[safe_leaf_ids] - item_embeddings
    else:
        raw_direction = torch.randn(
            item_embeddings.shape,
            dtype=item_embeddings.dtype,
            device=item_embeddings.device,
            generator=generator,
        )
    direction = _normalized_direction(raw_direction, delta)
    direction = direction * valid_mask.unsqueeze(1).to(direction.dtype)
    view_a = item_embeddings + epsilon_a * direction
    view_b = item_embeddings + epsilon_b * direction
    return view_a, view_b, {
        "direction": direction,
        "epsilon_a": epsilon_a,
        "epsilon_b": epsilon_b,
        "valid_mask": valid_mask,
    }


def create_mixture_item_views(
    item_embeddings,
    leaf_ids,
    parent_ids,
    valid_mask,
    leaf_prototypes,
    parent_prototypes,
    alpha=0.5,
    epsilon_max=0.1,
    delta=1e-8,
    generator=None,
):
    """Create V2 views from a fixed convex mixture of leaf/parent directions."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("mixture alpha must be in [0, 1]")
    if float(epsilon_max) < 0.0 or float(delta) <= 0.0:
        raise ValueError("invalid mixture augmentation magnitude")
    valid_mask = valid_mask.bool() & (leaf_ids >= 0) & (parent_ids >= 0)
    safe_leaf = leaf_ids.clamp_min(0).long()
    safe_parent = parent_ids.clamp_min(0).long()
    leaf_direction = _normalized_direction(
        leaf_prototypes[safe_leaf] - item_embeddings, delta
    )
    parent_direction = _normalized_direction(
        parent_prototypes[safe_parent] - item_embeddings, delta
    )
    direction = alpha * leaf_direction + (1.0 - alpha) * parent_direction
    direction = direction * valid_mask.unsqueeze(1).to(direction.dtype)
    epsilon_a = torch.rand(
        (item_embeddings.shape[0], 1), dtype=item_embeddings.dtype,
        device=item_embeddings.device, generator=generator,
    ) * float(epsilon_max)
    epsilon_b = torch.rand(
        (item_embeddings.shape[0], 1), dtype=item_embeddings.dtype,
        device=item_embeddings.device, generator=generator,
    ) * float(epsilon_max)
    return item_embeddings + epsilon_a * direction, item_embeddings + epsilon_b * direction, {
        "direction": direction,
        "leaf_direction": leaf_direction,
        "parent_direction": parent_direction,
        "epsilon_a": epsilon_a,
        "epsilon_b": epsilon_b,
        "valid_mask": valid_mask,
        "mixture_alpha": alpha,
    }


def degree_epsilon_scale(item_degree, gamma_cold=1.5, gamma_warm=0.5,
                          near_cold_max=5, long_tail_max=10):
    """Per-item epsilon_max multiplier from train degree.

    Bucket boundaries default to the same thresholds already used for group
    evaluation (utility/utility_train/group_evaluator.py GROUP_DEFINITIONS):
    degree in [1, near_cold_max] -> gamma_cold (amplify); degree in
    (near_cold_max, long_tail_max] -> 1.0 (neutral anchor bucket); degree >
    long_tail_max -> gamma_warm (dampen).

    Merged in from the former utility.utility_function.taxonomy_contrastive_v2
    (2026-08-22, project consolidation -- pure code relocation, no behavior
    change) so every taxonomy-CL primitive lives in one module.
    """
    if gamma_cold < 1.0:
        raise ValueError("gamma_cold must be >= 1.0")
    if not 0.0 < gamma_warm <= 1.0:
        raise ValueError("gamma_warm must be in (0, 1]")
    if near_cold_max <= 0 or long_tail_max <= near_cold_max:
        raise ValueError("invalid degree bucket boundaries")
    return torch.where(
        item_degree <= near_cold_max,
        torch.full_like(item_degree, gamma_cold, dtype=torch.float32),
        torch.where(
            item_degree <= long_tail_max,
            torch.ones_like(item_degree, dtype=torch.float32),
            torch.full_like(item_degree, gamma_warm, dtype=torch.float32),
        ),
    )


def create_item_views_adaptive(
    item_embeddings,
    leaf_ids,
    valid_mask,
    prototypes,
    item_degree,
    epsilon_max=0.1,
    delta=1e-8,
    direction_mode="taxonomy",
    gamma_cold=1.5,
    gamma_warm=0.5,
    near_cold_max=5,
    long_tail_max=10,
    generator=None,
):
    """Independent-epsilon, shared-direction views with degree-adaptive
    magnitude. Reduces exactly to create_item_views when gamma_cold ==
    gamma_warm == 1.0. Merged in from taxonomy_contrastive_v2 (see
    degree_epsilon_scale docstring above)."""
    if float(epsilon_max) < 0.0:
        raise ValueError("epsilon_max must be non-negative")
    if float(delta) <= 0.0:
        raise ValueError("delta must be positive")
    if direction_mode not in {"taxonomy", "random"}:
        raise ValueError("direction_mode must be taxonomy or random")
    if item_degree.shape[0] != item_embeddings.shape[0]:
        raise ValueError("item_degree must align with item_embeddings")
    valid_mask = valid_mask.bool() & (leaf_ids >= 0)

    scale = degree_epsilon_scale(
        item_degree, gamma_cold=gamma_cold, gamma_warm=gamma_warm,
        near_cold_max=near_cold_max, long_tail_max=long_tail_max,
    ).to(item_embeddings.dtype).unsqueeze(1)
    epsilon_max_i = float(epsilon_max) * scale

    epsilon_a = torch.rand(
        (item_embeddings.shape[0], 1),
        dtype=item_embeddings.dtype,
        device=item_embeddings.device,
        generator=generator,
    ) * epsilon_max_i
    epsilon_b = torch.rand(
        (item_embeddings.shape[0], 1),
        dtype=item_embeddings.dtype,
        device=item_embeddings.device,
        generator=generator,
    ) * epsilon_max_i

    if direction_mode == "taxonomy":
        safe_leaf_ids = leaf_ids.clamp_min(0).long()
        raw_direction = prototypes[safe_leaf_ids] - item_embeddings
    else:
        raw_direction = torch.randn(
            item_embeddings.shape,
            dtype=item_embeddings.dtype,
            device=item_embeddings.device,
            generator=generator,
        )
    direction = _normalized_direction(raw_direction, delta)
    direction = direction * valid_mask.unsqueeze(1).to(direction.dtype)
    view_a = item_embeddings + epsilon_a * direction
    view_b = item_embeddings + epsilon_b * direction
    return view_a, view_b, {
        "direction": direction,
        "epsilon_a": epsilon_a,
        "epsilon_b": epsilon_b,
        "epsilon_max_i": epsilon_max_i,
        "valid_mask": valid_mask,
    }


def leaf_aware_info_nce(view_a, view_b, leaf_ids, temperature=0.2,
                         same_leaf_weight=0.05, symmetric=False):
    """InfoNCE with same-leaf items reweighted as soft positives instead of
    hard negatives. Degenerates exactly to item_level_info_nce's loss value
    when same_leaf_weight == 0. Merged in from taxonomy_contrastive_v2 (see
    degree_epsilon_scale docstring above)."""
    if view_a.shape != view_b.shape or view_a.ndim != 2:
        raise ValueError("InfoNCE views must be equal-shaped matrices")
    if view_a.shape[0] != leaf_ids.numel():
        raise ValueError("leaf_ids must match views")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= float(same_leaf_weight) <= 1.0:
        raise ValueError("same_leaf_weight must be in [0, 1]")
    if view_a.shape[0] < 2:
        return (view_a.sum() + view_b.sum()) * 0.0

    leaves = leaf_ids.long()
    diagonal = torch.eye(leaves.numel(), dtype=torch.bool, device=leaves.device)
    same_leaf = (leaves[:, None] == leaves[None, :]) & ~diagonal

    def _asymmetric_loss(anchor, other):
        logits = F.normalize(anchor, dim=1) @ F.normalize(other, dim=1).transpose(0, 1)
        logits = logits / float(temperature)
        numerator = logits.masked_fill(~diagonal, -torch.inf)
        if float(same_leaf_weight) > 0.0:
            weak = logits + torch.log(
                torch.as_tensor(float(same_leaf_weight), device=logits.device)
            )
            numerator = torch.logaddexp(
                numerator, weak.masked_fill(~same_leaf, -torch.inf)
            )
        per_anchor = torch.logsumexp(logits, dim=1) - torch.logsumexp(numerator, dim=1)
        return per_anchor.mean()

    loss_ab = _asymmetric_loss(view_a, view_b)
    if not symmetric:
        return loss_ab
    return 0.5 * (loss_ab + _asymmetric_loss(view_b, view_a))


def item_level_info_nce(view_a, view_b, temperature=0.2, symmetric=False):
    """Asymmetric A->B InfoNCE by default, with diagonal positive labels."""
    if view_a.shape != view_b.shape or view_a.ndim != 2:
        raise ValueError("InfoNCE views must be equal-shaped matrices")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if view_a.shape[0] < 2:
        return (view_a.sum() + view_b.sum()) * 0.0
    normalized_a = F.normalize(view_a, dim=1)
    normalized_b = F.normalize(view_b, dim=1)
    logits_ab = normalized_a @ normalized_b.transpose(0, 1)
    logits_ab = logits_ab / float(temperature)
    labels = torch.arange(view_a.shape[0], device=view_a.device)
    loss_ab = F.cross_entropy(logits_ab, labels, reduction="mean")
    if not symmetric:
        return loss_ab
    logits_ba = normalized_b @ normalized_a.transpose(0, 1)
    logits_ba = logits_ba / float(temperature)
    return 0.5 * (
        loss_ab + F.cross_entropy(logits_ba, labels, reduction="mean")
    )
