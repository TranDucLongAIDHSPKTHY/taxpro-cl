"""TaxPro-CL: SimGCL backbone with taxonomy-guided item-view perturbation.

Keeps SimGCL's exact two-view aggregation structure (layer-0 excluded from
the mean, perturbation applied independently after each GCN layer, two
independent perturbed forward passes per batch for view_a/view_b) but
replaces the perturbation *source* on the item side: instead of
`sign(embedding) * normalize(random_noise) * epsilon` (SimGCL), it uses
`normalize(prototype[leaf] - embedding) * epsilon` (taxonomy direction),
where `prototype[leaf]` is an EMA over that leaf's item embeddings,
updated once per training step (stop-gradient) before that step's two
perturbed passes are computed. The per-item epsilon budget is divided by
the GCN layer count before being applied at each layer, so cumulative
displacement across layers stays bounded by `epsilon_max` rather than
scaling with layer count; `degree_epsilon_scale` additionally makes this
per-item magnitude degree-adaptive (`use_adaptive_epsilon`), and
`gamma_cold`/`gamma_warm` set the near-cold/warm scaling factors.

Configuration surface (see `configure/TaxPro-CL.txt` for the values used
in the main reported configuration):
- `direction_source` ("prototype" is the taxonomy-guided mechanism this
  paper evaluates); `prototype_mode` ("leaf" for leaf-level prototypes,
  "mixture" for a leaf/parent convex blend, used only as a sensitivity
  check, not the main configuration).
- `temperature` (item-side InfoNCE) and `temperature_user` (user-side,
  independently tunable so the two do not have to share one value).
- `isotropic_blend` blends a small SimGCL-style isotropic component into
  the unit direction before epsilon scaling, drawn fresh per (layer,
  view) call; default 0.0 (pure taxonomy direction). Used only as a
  sensitivity check in this paper (main configuration keeps it at 0.0).
- `same_leaf_weight` (beta) generalizes the item InfoNCE objective to
  treat other same-leaf batch items as soft positives; default 0.0 (main
  configuration uses the one-positive form). See
  `utility/utility_function/taxonomy_contrastive.py`.
- `use_user_ssl` adds a SimGCL-identical isotropic perturbation and
  InfoNCE term on the user side (on by default in the main configuration,
  since the reported comparison inherits SimGCL's own both-sides-SSL
  design); `warm_start_epochs` gates both perturbed passes behind an
  initial BPR-only phase.

`models/SimGCL.py` and `configure/SimGCL.txt` are not touched by this file;
SimGCL remains the independent baseline used for comparison throughout the
paper.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import utility.utility_data.data_graph
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.taxpro_trainer as taxpro_trainer
from config_path.config_path import evaluation_protocol_dir
from utility.utility_data.taxonomy import load_taxonomy, read_json, sha256_file
from utility.utility_function.taxonomy_contrastive import (
    PrototypeBank,
    degree_epsilon_scale,
    item_level_info_nce,
    leaf_aware_info_nce,
)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


def _normalized_direction(vector, delta):
    return vector / (vector.norm(dim=1, keepdim=True) + float(delta))


def _build_leaf_peer_index(item_to_leaf_id, valid_mask):
    """Precompute a vectorized structure for sampling, per item, a different
    item that shares the same leaf -- O(1) gather at call time, no Python
    loop over leaves. Items with invalid taxonomy or singleton leaves get a
    harmless self-referencing entry (masked out by valid_mask downstream)."""
    num_items = item_to_leaf_id.shape[0]
    safe_leaf = item_to_leaf_id.clamp_min(0).long()
    grouped_leaf = torch.where(
        valid_mask, safe_leaf, torch.full_like(safe_leaf, -1)
    )
    # Invalid items get their own leaf id (-1 - item_index, all distinct) so
    # they form singleton groups and never get selected as anyone's peer.
    grouped_leaf = torch.where(
        valid_mask,
        grouped_leaf,
        -1 - torch.arange(num_items, dtype=grouped_leaf.dtype),
    )
    sort_idx = torch.argsort(grouped_leaf, stable=True)
    sorted_leaf = grouped_leaf[sort_idx]
    _, counts = torch.unique_consecutive(sorted_leaf, return_counts=True)
    group_start_per_run = torch.cumsum(counts, dim=0) - counts
    run_start_at_position = torch.repeat_interleave(group_start_per_run, counts)
    run_size_at_position = torch.repeat_interleave(counts, counts)

    item_group_start = torch.zeros(num_items, dtype=torch.long)
    item_group_size = torch.ones(num_items, dtype=torch.long)
    item_rank = torch.zeros(num_items, dtype=torch.long)
    item_group_start[sort_idx] = run_start_at_position
    item_group_size[sort_idx] = run_size_at_position
    item_rank[sort_idx] = torch.arange(num_items) - run_start_at_position
    return sort_idx, item_group_start, item_group_size, item_rank


class TaxProCLImproved(nn.Module):
    """SimGCL-structured backbone (layer-0 excluded, per-layer independent
    perturbation) with taxonomy-direction perturbation instead of random
    noise, item side. User-side contrastive loss (SimGCL's original
    isotropic-noise InfoNCE, unchanged) is a separately config-gated branch
    (use_user_ssl) -- ON in every reported main-config result in the paper
    (Sec 3.4/4.4), not "item-only".

    Self-contained: the base construction, `initialize_prototypes`,
    `get_rating_for_test` and `checkpoint_metadata` logic live in this class
    (git history keeps the earlier base class it was inlined from)."""

    def __init__(self, config, dataset, device):
        super().__init__()
        self.config = dict(config)
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(config["reg_lambda"])
        self.ssl_lambda = float(config["ssl_lambda"])
        self.temperature = float(config["temperature"])
        self.epsilon_max = float(config["epsilon_max"])
        self.delta = float(config["delta"])
        self.warm_start_epochs = int(config["warm_start_epochs"])
        self.augmentation_direction = str(config["augmentation_direction"])
        self.taxonomy_granularity = str(config.get("taxonomy_granularity", "leaf"))
        self.prototype_mode = str(
            config.get("prototype_mode", self.taxonomy_granularity)
        )
        self.prototype_update = str(config.get("prototype_update", "ema"))
        self.prototype_init_scope = str(config.get("prototype_init_scope", "all_valid"))
        self.mixture_alpha = float(config.get("mixture_alpha", 0.5))
        self.symmetric_info_nce = _as_bool(config["symmetric_info_nce"])
        # Three sensitivity-check flags for the prototype-construction ablation
        # ("unique-item EMA"/"leave-one-out" variants, plus an asymmetric-view
        # variant), each defaulting to the prior behavior, so every earlier
        # run's config stays byte-for-byte reproducible when none are set.
        self.prototype_weighting = str(config.get("prototype_weighting", "interaction"))
        self.prototype_leave_one_out = _as_bool(config.get("prototype_leave_one_out", False))
        self.asymmetric_view_direction = _as_bool(config.get("asymmetric_view_direction", False))
        if self.reg_lambda < 0.0 or self.ssl_lambda < 0.0:
            raise ValueError("Loss weights must be non-negative")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.epsilon_max < 0.0:
            raise ValueError("epsilon_max must be non-negative")
        if self.delta <= 0.0:
            raise ValueError("delta must be positive")
        if self.warm_start_epochs < 0:
            raise ValueError("warm_start_epochs must be non-negative")
        if self.augmentation_direction not in {"taxonomy", "random"}:
            raise ValueError("augmentation_direction must be taxonomy or random")
        if self.prototype_mode not in {"leaf", "parent", "mixture"}:
            raise ValueError("prototype_mode must be leaf, parent, or mixture")
        if self.prototype_update not in {"ema", "fixed"}:
            raise ValueError("prototype_update must be ema or fixed")
        if self.prototype_init_scope not in {"all_valid", "warm_observed"}:
            raise ValueError("prototype_init_scope must be all_valid or warm_observed")
        if not 0.0 <= self.mixture_alpha <= 1.0:
            raise ValueError("mixture_alpha must be in [0, 1]")
        if self.prototype_mode in {"parent", "mixture"} and str(config["dataset"]) != "amazon-book":
            raise ValueError("Parent and mixture variants are available only for amazon-book")
        if str(config["unknown_taxonomy_policy"]) != "exclude_cl":
            raise ValueError("TaxPro-CL requires unknown_taxonomy_policy=exclude_cl")
        if self.prototype_weighting not in {"interaction", "leaf_uniform"}:
            raise ValueError("prototype_weighting must be interaction or leaf_uniform")

        self.user_embedding = nn.Embedding(
            dataset.num_users, int(config["embedding_size"])
        )
        self.item_embedding = nn.Embedding(
            dataset.num_items, int(config["embedding_size"])
        )
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        graph = utility.utility_data.data_graph.sparse_adjacency_matrix(dataset)
        graph = tools.convert_sp_mat_to_sp_tensor(graph)
        self.Graph = graph.coalesce().to(device)
        self.activation = nn.Sigmoid()

        primary_granularity = "parent" if self.prototype_mode == "parent" else "leaf"
        taxonomy = load_taxonomy(
            str(config["dataset"]),
            str(config["taxonomy_policy"]),
            primary_granularity,
        )
        if taxonomy.num_items != dataset.num_items:
            raise ValueError(
                "Taxonomy catalog size {} != dataset items {}".format(
                    taxonomy.num_items, dataset.num_items
                )
            )
        tensors = taxonomy.as_torch()
        observed = torch.as_tensor(
            dataset.user_item_net.getnnz(axis=0) > 0, dtype=torch.bool
        )
        tensors["train_observed_mask"] = observed
        tensors["valid_train_mask"] = tensors["valid_taxonomy_mask"] & observed
        self.register_buffer("item_to_leaf_id", tensors["item_to_leaf_id"])
        self.register_buffer("valid_taxonomy_mask", tensors["valid_taxonomy_mask"])
        self.register_buffer("train_observed_mask", tensors["train_observed_mask"])
        self.register_buffer("valid_train_mask", tensors["valid_train_mask"])
        # A6 ablation ("EMA on/off; warm-mean vs all-item"):
        # tracks which items actually appeared as a positive sample during the
        # warm_start window, so initialize_prototypes() can optionally average
        # only those ("warm_observed") instead of every valid-taxonomy item
        # regardless of when it was seen ("all_valid", the pre-existing/default
        # behavior -- unchanged, every prior run stays byte-identical).
        # persistent=False: pure training-time tracking, always starts at
        # all-False and gets rebuilt from scratch during warm_start -- must
        # NOT be part of state_dict(), otherwise every checkpoint saved after
        # this buffer was added would require it, breaking load_state_dict()
        # for any older checkpoint trained before this buffer existed.
        self.register_buffer(
            "warm_observed_mask",
            torch.zeros(dataset.num_items, dtype=torch.bool),
            persistent=False,
        )
        self.prototype_bank = PrototypeBank(
            taxonomy.num_prototypes,
            int(config["embedding_size"]),
            mu=float(config["mu"]),
        )
        self.parent_prototype_bank = None
        if self.prototype_mode in ("mixture", "parent"):
            parent_taxonomy = load_taxonomy(
                str(config["dataset"]), str(config["taxonomy_policy"]), "parent"
            )
            parent_tensors = parent_taxonomy.as_torch()
            parent_tensors["valid_train_mask"] = (
                parent_tensors["valid_taxonomy_mask"] & observed
            )
            self.register_buffer(
                "item_to_parent_id", parent_tensors["item_to_leaf_id"]
            )
            self.register_buffer(
                "valid_parent_train_mask", parent_tensors["valid_train_mask"]
            )
            self.parent_prototype_bank = PrototypeBank(
                parent_taxonomy.num_prototypes,
                int(config["embedding_size"]),
                mu=float(config["mu"]),
            )
            self.parent_taxonomy_hash = parent_taxonomy.taxonomy_hash
        else:
            self.parent_taxonomy_hash = None
        self.taxonomy_hash = taxonomy.taxonomy_hash
        self.taxonomy_protocol_version = taxonomy.manifest["protocol_version"]
        self.taxonomy_statistics = taxonomy.statistics
        protocol_root = Path(str(config.get(
            "evaluation_protocol_path",
            evaluation_protocol_dir(str(config["dataset"])).parent,
        )))
        if not protocol_root.is_absolute():
            protocol_root = Path(__file__).resolve().parents[1] / protocol_root
        evaluation_manifest_path = protocol_root / str(config["dataset"]) / "manifest.json"
        self.evaluation_protocol_hash = sha256_file(evaluation_manifest_path)
        self.split_hashes = read_json(evaluation_manifest_path)["source_hashes"]
        self.last_batch_statistics = {}

        if self.prototype_mode not in ("leaf", "mixture", "parent"):
            raise ValueError(
                "TaxPro-CL currently supports prototype_mode=leaf, mixture, or parent only"
            )

        self.use_adaptive_epsilon = _as_bool(config.get("use_adaptive_epsilon", True))
        self.use_leaf_aware_infonce = _as_bool(
            config.get("use_leaf_aware_infonce", True)
        )
        self.gamma_cold = float(config.get("gamma_cold", 1.5))
        self.gamma_warm = float(config.get("gamma_warm", 0.5))
        self.same_leaf_weight = float(config.get("same_leaf_weight", 0.05))
        self.degree_near_cold_max = int(config.get("degree_near_cold_max", 5))
        self.degree_long_tail_max = int(config.get("degree_long_tail_max", 10))
        if self.gamma_cold < 1.0:
            raise ValueError("gamma_cold must be >= 1.0")
        if not 0.0 < self.gamma_warm <= 1.0:
            raise ValueError("gamma_warm must be in (0, 1]")
        if not 0.0 <= self.same_leaf_weight <= 1.0:
            raise ValueError("same_leaf_weight must be in [0, 1]")
        if (
            self.degree_near_cold_max <= 0
            or self.degree_long_tail_max <= self.degree_near_cold_max
        ):
            raise ValueError("invalid degree bucket boundaries")

        # Canonical train-degree source: the same file utility.utility_data
        # .taxonomy.load_taxonomy() reads to build train_observed_mask, so
        # the (A) buckets stay consistent with group_evaluator.py's
        # near_cold/long_tail/warm thresholds.
        degree_payload = read_json(
            evaluation_protocol_dir(str(config["dataset"])) / "item_degrees_train.json"
        )
        degrees = torch.as_tensor(degree_payload["degrees"], dtype=torch.long)
        if degrees.shape[0] != dataset.num_items:
            raise ValueError(
                "item_degrees_train.json does not match dataset catalog size"
            )
        self.register_buffer("train_item_degree", degrees)

        self.use_user_ssl = _as_bool(config.get("use_user_ssl", False))
        self.epsilon_user = float(config.get("epsilon_user", 0.05))
        self.ssl_lambda_user = float(config.get("ssl_lambda_user", 0.5))
        if self.epsilon_user < 0.0:
            raise ValueError("epsilon_user must be non-negative")
        if self.ssl_lambda_user < 0.0:
            raise ValueError("ssl_lambda_user must be non-negative")

        # Decoupled from the item-side `temperature` so item-side sharpening
        # (validated below) does not silently drag the user-side SSL loss away
        # from SimGCL's own tuned value. Defaults to 0.2 (SimGCL's own
        # temperature), which keeps configurations that predate this option
        # reproducible.
        self.temperature_user = float(config.get("temperature_user", 0.2))
        if self.temperature_user <= 0.0:
            raise ValueError("temperature_user must be positive")

        self.isotropic_blend = float(config.get("isotropic_blend", 0.0))
        if not 0.0 <= self.isotropic_blend <= 1.0:
            raise ValueError("isotropic_blend must be in [0, 1]")

        self.direction_source = str(config.get("direction_source", "prototype"))
        if self.direction_source not in ("prototype", "peer"):
            raise ValueError("direction_source must be prototype or peer")
        if self.direction_source == "peer":
            sort_idx, group_start, group_size, item_rank = _build_leaf_peer_index(
                self.item_to_leaf_id, self.valid_train_mask
            )
            self.register_buffer("leaf_peer_sorted_items", sort_idx)
            self.register_buffer("leaf_peer_group_start", group_start)
            self.register_buffer("leaf_peer_group_size", group_size)
            self.register_buffer("leaf_peer_item_rank", item_rank)

    @torch.no_grad()
    def initialize_prototypes(self, epoch):
        if bool(self.prototype_bank.initialized.item()):
            raise RuntimeError("Prototype initialization may run only once")
        was_training = self.training
        self.eval()
        _, item_embeddings = self.aggregate()
        init_mask = self.valid_train_mask
        if self.prototype_init_scope == "warm_observed":
            init_mask = init_mask & self.warm_observed_mask
        evidence = self.prototype_bank.initialize(
            item_embeddings,
            self.item_to_leaf_id,
            init_mask,
            epoch=epoch,
        )
        if self.parent_prototype_bank is not None:
            parent_init_mask = self.valid_parent_train_mask
            if self.prototype_init_scope == "warm_observed":
                parent_init_mask = parent_init_mask & self.warm_observed_mask
            evidence["parent"] = self.parent_prototype_bank.initialize(
                item_embeddings,
                self.item_to_parent_id,
                parent_init_mask,
                epoch=epoch,
            )
        if was_training:
            self.train()
        return evidence

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate()
        user_embeddings = all_user_embeddings[user.long()]
        return self.activation(user_embeddings @ all_item_embeddings.transpose(0, 1))

    def aggregate(self, perturbed=False, view_id=None):
        """SimGCL-style aggregation: layer-0 excluded from the mean, each of
        GCN_layer propagation steps optionally perturbed independently.
        Unperturbed calls (perturbed=False, the default) are byte-identical
        in structure to SimGCL's clean pass and are used for BPR, prototype
        initialization and evaluation -- exactly like the locked baseline's
        single aggregate() was used for everything.

        view_id (used with asymmetric_view_direction): identifies
        which of the two perturbed passes this call is (0 or 1) so
        _perturb_items can force view 1's direction to random/isotropic when
        asymmetric_view_direction=True. None/unused when that flag is off."""
        all_embedding = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight]
        )
        embeddings = []
        gcn_layers = int(self.config["GCN_layer"])
        apply_item_perturbation = perturbed and bool(self.prototype_bank.initialized.item())
        if apply_item_perturbation and self.prototype_mode == "mixture":
            apply_item_perturbation = bool(self.parent_prototype_bank.initialized.item())
        apply_user_perturbation = perturbed and self.use_user_ssl
        for _ in range(gcn_layers):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            if apply_user_perturbation:
                all_embedding = self._perturb_users(all_embedding)
            if apply_item_perturbation:
                all_embedding = self._perturb_items(all_embedding, gcn_layers, view_id=view_id)
            embeddings.append(all_embedding)
        final_embeddings = torch.stack(embeddings, dim=1).mean(dim=1)
        return torch.split(
            final_embeddings, [self.dataset.num_users, self.dataset.num_items]
        )

    def _perturb_users(self, all_embedding):
        """SimGCL-identical isotropic noise
        (`sign(e) * normalize(noise) * epsilon`, undivided per layer, exactly
        matching models/SimGCL.py) applied only to the user slice. Users
        carry no taxonomy attribute, so no taxonomy-derived direction exists
        for them -- this is a separate, additive branch that recovers the
        user-side half of SimGCL's SSL signal (SimGCL sums user_ssl_loss +
        item_ssl_loss; TaxPro-CL previously had zero user-side term). Does
        not touch the item taxonomy augmenter, which stays 100%
        taxonomy-derived. Gated behind use_user_ssl (default False) so every
        prior run's config stays byte-for-byte reproducible."""
        users, items = torch.split(
            all_embedding, [self.dataset.num_users, self.dataset.num_items]
        )
        noise = torch.rand_like(users)
        users = users + torch.sign(users) * torch.nn.functional.normalize(
            noise, dim=-1
        ) * self.epsilon_user
        return torch.cat([users, items])

    def _perturb_items(self, all_embedding, gcn_layers, view_id=None):
        """Taxonomy-direction perturbation applied to the item slice of one
        GCN layer's output. epsilon_max is divided by gcn_layers so the
        cumulative displacement across all layers stays on the same scale
        as the single-shot v1/v2 design (mitigates the over-smoothing risk
        the source PDF itself flags for repeated same-direction pushes).

        view_id + asymmetric_view_direction: with
        augmentation_direction="taxonomy" (the main configuration), both
        perturbed passes previously received the SAME deterministic
        direction (Eq. 1), differing only in the scalar epsilon -- near-
        colinear views, near-zero InfoNCE positive gradient (see A4
        analysis). When asymmetric_view_direction=True, view 1 is forced to
        the random/isotropic direction (same formula as
        augmentation_direction="random" below) regardless of the configured
        augmentation_direction, guaranteeing non-colinear views by
        construction rather than by noisy isotropic_blend mixing (which was
        tried and made Amazon-Book worse, not better)."""
        users, items = torch.split(
            all_embedding, [self.dataset.num_users, self.dataset.num_items]
        )
        effective_direction = self.augmentation_direction
        if self.asymmetric_view_direction and view_id == 1:
            effective_direction = "random"
        if effective_direction == "random":
            # A2 control: a genuine isotropic random direction, independent
            # of taxonomy/prototype -- reuses
            # the exact SimGCL-identical recipe already used for
            # isotropic_blend (sign(e) * normalize(noise), fresh per layer
            # per view) instead of the prototype-relative direction below.
            # valid_mask stays self.valid_train_mask so V0/V2 perturb the
            # same item population as V1/V3 (Eq. (1) taxonomy variants);
            # only the direction formula differs, isolating that one factor.
            noise = torch.rand_like(items)
            direction = torch.sign(items) * torch.nn.functional.normalize(
                noise, dim=-1
            )
            valid_mask = self.valid_train_mask
        else:
            safe_leaf = self.item_to_leaf_id.clamp_min(0).long()
            if self.direction_source == "peer":
                # peer variant: stochastic same-leaf peer target instead of the fixed EMA
                # prototype -- independently resampled at every (layer, view)
                # call, so the two views get genuinely different directions, not
                # just different magnitudes along one shared axis. Still 100%
                # taxonomy-derived (peer pool is leaf-membership only) and still
                # single-level leaf taxonomy -- no parent/multi-level signal, no
                # loss-term change, no gating: a different concrete
                # instantiation of the same "taxonomy-guided view generator"
                # mechanism, not a different one.
                group_size = self.leaf_peer_group_size
                safe_group_size = group_size.clamp_min(2)
                offset = torch.randint(
                    low=1,
                    high=1_000_000_007,
                    size=(items.shape[0],),
                    device=items.device,
                ) % (safe_group_size - 1).clamp_min(1) + 1
                peer_rank = (self.leaf_peer_item_rank + offset) % safe_group_size
                # Singleton groups use a clamped virtual size of 2 purely so the
                # modulo above is well-defined; that can point one slot past the
                # group's single real entry, so clamp the resulting flat index
                # into range. These items are masked to zero displacement below
                # regardless (no real peer exists), so the gathered value here
                # is never actually used.
                peer_position = (self.leaf_peer_group_start + peer_rank).clamp(
                    0, items.shape[0] - 1
                )
                peer_item_idx = self.leaf_peer_sorted_items[peer_position]
                peer_embedding = items[peer_item_idx]
                leaf_direction = _normalized_direction(peer_embedding - items, self.delta)
            else:
                target_prototype = self.prototype_bank.prototypes[safe_leaf]
                if self.prototype_leave_one_out:
                    # A3 "leave-one-out" variant: p_l is a
                    # mean over N_l valid items in the leaf (support, fixed at
                    # init -- see PrototypeBank.initialize); an item with a
                    # small leaf can dominate its own prototype (A3 measured
                    # this directly: leaf-size-1/2 items have median
                    # ||p_l - e_i|| of 0.09-0.15 vs 1.5-2.4 for large leaves,
                    # i.e. near self-reference). Closed-form leave-one-out:
                    # p_l^(-i) = (N_l * p_l - e_i) / (N_l - 1). Leaves with
                    # N_l < 2 have no leave-one-out target at all; those items
                    # are excluded via valid_mask below, not clamped here.
                    leaf_support = self.prototype_bank.support[safe_leaf].to(
                        items.dtype
                    ).unsqueeze(1)
                    safe_support = leaf_support.clamp_min(2.0)
                    target_prototype = (
                        safe_support * target_prototype - items
                    ) / (safe_support - 1.0)
                leaf_direction = _normalized_direction(
                    target_prototype - items, self.delta
                )
            if self.prototype_mode == "mixture":
                # Fixed convex mixture of leaf and parent prototype directions, mirroring
                # the locked baseline's create_mixture_item_views() formula exactly
                # -- alpha * d_leaf + (1-alpha) * d_parent, each normalized before
                # blending. Still 100% taxonomy-derived, no isotropic component.
                safe_parent = self.item_to_parent_id.clamp_min(0).long()
                parent_direction = _normalized_direction(
                    self.parent_prototype_bank.prototypes[safe_parent] - items, self.delta
                )
                direction = (
                    self.mixture_alpha * leaf_direction
                    + (1.0 - self.mixture_alpha) * parent_direction
                )
                valid_mask = self.valid_train_mask & self.valid_parent_train_mask
            elif self.prototype_mode == "parent":
                # A5 ablation ("leaf vs parent granularity"):
                # pure parent-level direction, no leaf blending. Mathematically
                # identical to prototype_mode=mixture with mixture_alpha=0.0
                # (0*leaf + 1*parent = parent) -- that combination was validated
                # first (see ablation-A5-mixture-alpha0.0 run history) before this
                # explicit, self-documenting mode was added so config_resolved.json
                # reads "prototype_mode": "parent" directly instead of requiring a
                # reader to know the mixture_alpha=0.0 equivalence.
                safe_parent = self.item_to_parent_id.clamp_min(0).long()
                direction = _normalized_direction(
                    self.parent_prototype_bank.prototypes[safe_parent] - items, self.delta
                )
                valid_mask = self.valid_train_mask & self.valid_parent_train_mask
            else:
                direction = leaf_direction
                valid_mask = self.valid_train_mask
                if self.prototype_leave_one_out:
                    valid_mask = valid_mask & (
                        self.prototype_bank.support[safe_leaf] >= 2
                    )
            if self.direction_source == "peer":
                # A leaf with only one valid member has no distinct peer to
                # offer -- exclude those items rather than silently reusing a
                # neighboring leaf's item from the sort-order fallback.
                valid_mask = valid_mask & (self.leaf_peer_group_size >= 2)
        if self.isotropic_blend > 0.0:
            # Mitigation for the two views becoming too close to collinear:
            # blends a small SimGCL-identical isotropic component (same
            # sign(e)*normalize(noise) recipe as _perturb_users/SimGCL.py,
            # drawn fresh per layer per view call) into the *unit direction*,
            # not added on top of it -- so the existing epsilon budget
            # (adaptive-epsilon, per-layer division, etc.) is untouched and
            # only the direction's composition changes. Only meaningfully
            # needed for direction_source="prototype" (the fixed-target case
            # that produces near-colinear views, cos~=0.925 between the two
            # views with this mechanism); left general/orthogonal rather than
            # special-cased so it composes with "peer" too if ever wanted.
            # Explicitly NOT "100% taxonomy-derived" on the item side once
            # isotropic_blend>0 -- a deliberate, disclosed departure from that
            # standing constraint, gated behind isotropic_blend defaulting to
            # 0.0 so every prior run's config stays byte-for-byte
            # reproducible.
            noise = torch.rand_like(items)
            isotropic_direction = torch.sign(items) * torch.nn.functional.normalize(
                noise, dim=-1
            )
            direction = _normalized_direction(
                (1.0 - self.isotropic_blend) * direction
                + self.isotropic_blend * isotropic_direction,
                self.delta,
            )
        direction = direction * valid_mask.unsqueeze(1).to(direction.dtype)
        if self.use_adaptive_epsilon:
            scale = degree_epsilon_scale(
                self.train_item_degree,
                gamma_cold=self.gamma_cold,
                gamma_warm=self.gamma_warm,
                near_cold_max=self.degree_near_cold_max,
                long_tail_max=self.degree_long_tail_max,
            ).to(items.dtype).unsqueeze(1)
        else:
            scale = torch.ones((items.shape[0], 1), dtype=items.dtype, device=items.device)
        per_layer_epsilon_max = (self.epsilon_max / float(gcn_layers)) * scale
        epsilon = torch.rand(
            (items.shape[0], 1), dtype=items.dtype, device=items.device
        ) * per_layer_epsilon_max
        items = items + epsilon * direction
        return torch.cat([users, items])

    def forward(self, user, positive, negative, epoch):
        if self.prototype_init_scope == "warm_observed" and int(epoch) <= self.warm_start_epochs:
            with torch.no_grad():
                self.warm_observed_mask[positive.long()] = True
        all_user_embeddings, all_item_embeddings = self.aggregate(perturbed=False)
        user_embedding = all_user_embeddings[user.long()]
        positive_embedding = all_item_embeddings[positive.long()]
        negative_embedding = all_item_embeddings[negative.long()]

        bpr_loss = losses.get_bpr_loss(
            user_embedding, positive_embedding, negative_embedding
        )
        regularization = losses.get_reg_loss(
            self.user_embedding(user),
            self.item_embedding(positive),
            self.item_embedding(negative),
        )
        regularization = self.reg_lambda * regularization
        cl_item_loss = (bpr_loss + regularization) * 0.0
        user_ssl_loss = (bpr_loss + regularization) * 0.0
        ema_statistics = {
            "leaf_update_count": 0,
            "item_update_count": 0,
            "prototype_norm_mean": 0.0,
            "prototype_drift_mean": 0.0,
            "inactive_leaf_count": int(self.prototype_bank.prototypes.shape[0]),
            "invalid_taxonomy_item_count": 0,
            "cl_item_count": 0,
            "augmentation_norm_max": 0.0,
            "augmentation_direction_cosine_min": None,
            "identical_view_fraction": 0.0,
            "view_displacement_cosine_mean": None,
            "user_ssl_loss": 0.0,
        }

        if int(epoch) > self.warm_start_epochs:
            if not bool(self.prototype_bank.initialized.item()):
                raise RuntimeError("Joint phase requires initialized prototypes")
            if self.prototype_update == "ema":
                if self.prototype_weighting == "leaf_uniform":
                    ema_statistics.update(
                        self.prototype_bank.ema_update_snapshot(
                            all_item_embeddings, self.item_to_leaf_id,
                            self.valid_train_mask, epoch,
                        )
                    )
                else:
                    ema_statistics.update(
                        self.prototype_bank.ema_update(
                            all_item_embeddings, positive, self.item_to_leaf_id,
                            self.valid_train_mask,
                        )
                    )
                if self.prototype_mode in ("mixture", "parent"):
                    parent_stats = self.parent_prototype_bank.ema_update(
                        all_item_embeddings, positive, self.item_to_parent_id,
                        self.valid_parent_train_mask,
                    )
                    ema_statistics.update(
                        {"parent_" + key: value for key, value in parent_stats.items()}
                    )
            unique_positive = torch.unique(positive.long())
            valid = self.valid_train_mask[unique_positive]
            if self.prototype_mode in ("mixture", "parent"):
                valid = valid & self.valid_parent_train_mask[unique_positive]
            valid_items = unique_positive[valid]
            need_perturbed_pass = valid_items.numel() >= 1 or self.use_user_ssl
            if need_perturbed_pass:
                # Two independent perturbed passes through the full encoder,
                # exactly mirroring SimGCL's forward(): aggregate(True) is
                # called twice, each drawing fresh randomness at every layer.
                # Same two passes serve both the item taxonomy CL view (v3-v7)
                # and, when enabled, the v9 user-side SSL view -- no extra
                # forward cost for turning use_user_ssl on.
                user_embeddings_a, item_embeddings_a = self.aggregate(perturbed=True, view_id=0)
                user_embeddings_b, item_embeddings_b = self.aggregate(perturbed=True, view_id=1)

                if valid_items.numel() >= 1:
                    leaves = self.item_to_leaf_id[valid_items]
                    clean_items = all_item_embeddings[valid_items]
                    view_a = item_embeddings_a[valid_items]
                    view_b = item_embeddings_b[valid_items]

                    displacement_a = view_a - clean_items
                    displacement_b = view_b - clean_items
                    displacement = torch.cat(
                        (displacement_a.norm(dim=1), displacement_b.norm(dim=1))
                    )
                    identical_views = torch.all(view_a == view_b, dim=1)
                    # Empirical stand-in for a "cos(view A, view B)"
                    # diagnostic: with per-layer perturbation there
                    # is no longer one direction vector to compare, so this
                    # compares the two views' NET displacement from the
                    # clean embedding instead. High mean here (~0.925 in the
                    # near-colinear case) means the two views are near-
                    # colinear -- a weak-positive-pull failure mode;
                    # isotropic_blend>0 is meant to push this down.
                    nonzero_displacement = (
                        (displacement_a.norm(dim=1) > self.delta)
                        & (displacement_b.norm(dim=1) > self.delta)
                    )
                    if bool(nonzero_displacement.any().item()):
                        view_displacement_cosine_mean = float(
                            torch.nn.functional.cosine_similarity(
                                displacement_a[nonzero_displacement],
                                displacement_b[nonzero_displacement],
                                dim=1,
                            )
                            .detach()
                            .mean()
                            .item()
                        )
                    else:
                        view_displacement_cosine_mean = None
                    ema_statistics.update(
                        {
                            "cl_item_count": int(valid_items.numel()),
                            "augmentation_norm_max": float(
                                displacement.detach().max().item()
                            ),
                            # No single direction vector to report any more --
                            # direction is recomputed independently at each of
                            # the GCN_layer perturbation points.
                            "augmentation_direction_cosine_min": None,
                            "identical_view_fraction": float(
                                identical_views.float().mean().detach().item()
                            ),
                            "view_displacement_cosine_mean": view_displacement_cosine_mean,
                        }
                    )
                    if self.use_leaf_aware_infonce:
                        cl_item_loss = leaf_aware_info_nce(
                            view_a, view_b, leaves,
                            temperature=self.temperature,
                            same_leaf_weight=self.same_leaf_weight,
                            symmetric=self.symmetric_info_nce,
                        )
                    else:
                        cl_item_loss = item_level_info_nce(
                            view_a, view_b,
                            temperature=self.temperature,
                            symmetric=self.symmetric_info_nce,
                        )

                if self.use_user_ssl:
                    unique_user = torch.unique(user.long())
                    user_view_a = user_embeddings_a[unique_user]
                    user_view_b = user_embeddings_b[unique_user]
                    user_ssl_loss = losses.get_InfoNCE_loss(
                        user_view_a, user_view_b, self.temperature_user
                    )
                    ema_statistics["user_ssl_loss"] = float(
                        user_ssl_loss.detach().item()
                    )

        weighted_cl = self.ssl_lambda * cl_item_loss
        weighted_user_ssl = self.ssl_lambda_user * user_ssl_loss
        total = bpr_loss + regularization + weighted_cl + weighted_user_ssl
        self.last_batch_statistics = {
            **ema_statistics,
            "bpr_loss": float(bpr_loss.detach().item()),
            "reg_loss": float(regularization.detach().item()),
            "cl_item_loss": float(cl_item_loss.detach().item()),
            "weighted_user_ssl_loss": float(weighted_user_ssl.detach().item()),
            "total_loss": float(total.detach().item()),
            "joint_phase": bool(int(epoch) > self.warm_start_epochs),
            "use_adaptive_epsilon": self.use_adaptive_epsilon,
            "use_leaf_aware_infonce": self.use_leaf_aware_infonce,
            "use_user_ssl": self.use_user_ssl,
            "perturbation_structure": "per_layer_taxonomy_direction",
        }
        # taxpro_trainer.py accumulates exactly 3 loss slots (bpr/reg/cl) --
        # keep this list length fixed at 3, folding weighted_user_ssl into
        # the cl slot, exactly how models/SimGCL.py bundles
        # user_ssl_loss + item_ssl_loss into a single returned ssl_loss.
        return [bpr_loss, regularization, weighted_cl + weighted_user_ssl]

    def checkpoint_metadata(self):
        metadata = {
            "model": "TaxProCL",
            "dataset": str(self.config["dataset"]),
            "taxonomy_policy": str(self.config["taxonomy_policy"]),
            "taxonomy_granularity": self.taxonomy_granularity,
            "prototype_mode": self.prototype_mode,
            "prototype_update": self.prototype_update,
            "prototype_init_scope": self.prototype_init_scope,
            "mixture_alpha": self.mixture_alpha,
            "parent_taxonomy_hash": self.parent_taxonomy_hash,
            "taxonomy_hash": self.taxonomy_hash,
            "taxonomy_protocol_version": self.taxonomy_protocol_version,
            "evaluation_protocol_hash": self.evaluation_protocol_hash,
            "split_hashes": self.split_hashes,
            "prototype_initialized": bool(
                self.prototype_bank.initialized.detach().cpu().item()
            ),
            "prototype_init_epoch": int(
                self.prototype_bank.init_epoch.detach().cpu().item()
            ),
            "augmentation_direction": self.augmentation_direction,
            "unknown_taxonomy_policy": "exclude_cl",
            "symmetric_info_nce": self.symmetric_info_nce,
        }
        metadata.update({
            "model": "TaxPro-CL",
            "improvement": (
                "TaxPro-CL-isotropic-blend"
                if self.isotropic_blend > 0.0
                else "TaxPro-CL-user-ssl"
                if self.use_user_ssl
                else "TaxPro-CL-simgcl-backbone-peer-direction"
                if self.direction_source == "peer"
                else "TaxPro-CL-simgcl-backbone-mixture"
                if self.prototype_mode == "mixture"
                else "TaxPro-CL-simgcl-backbone"
            ),
            "perturbation_structure": "per_layer_taxonomy_direction",
            "direction_source": self.direction_source,
            "use_adaptive_epsilon": self.use_adaptive_epsilon,
            "use_leaf_aware_infonce": self.use_leaf_aware_infonce,
            "gamma_cold": self.gamma_cold,
            "gamma_warm": self.gamma_warm,
            "same_leaf_weight": self.same_leaf_weight,
            "degree_near_cold_max": self.degree_near_cold_max,
            "degree_long_tail_max": self.degree_long_tail_max,
            "use_user_ssl": self.use_user_ssl,
            "epsilon_user": self.epsilon_user,
            "ssl_lambda_user": self.ssl_lambda_user,
            "temperature_user": self.temperature_user,
            "isotropic_blend": self.isotropic_blend,
            "prototype_weighting": self.prototype_weighting,
            "prototype_leave_one_out": self.prototype_leave_one_out,
            "asymmetric_view_direction": self.asymmetric_view_direction,
        })
        return metadata

    def effective_config(self):
        # Readable, filtered view of the config that actually drives this run's
        # behavior -- filters out only mode-gated fields that are dead weight
        # for the current settings (e.g. mixture_alpha when prototype_mode !=
        # "mixture"), which makes reading a training.log harder than it needs
        # to be once a config is locked in for ablations. Infra/always-used
        # parameters (batch size, learning rate, worker count, ...) are NOT
        # dead weight -- they are genuinely read every run -- so they stay
        # here too instead of being silently dropped from the log.
        config = {
            "dataset_path": str(self.config["dataset_path"]),
            "evaluation_protocol_path": str(self.config["evaluation_protocol_path"]),
            "top_K": str(self.config["top_K"]),
            "selection_K": self.config["selection_K"],
            "training_epochs": self.config["training_epochs"],
            "early_stopping": self.config["early_stopping"],
            "interval": self.config["interval"],
            "embedding_size": self.config["embedding_size"],
            "batch_size": self.config["batch_size"],
            "test_batch_size": self.config["test_batch_size"],
            "num_worker": self.config["num_worker"],
            "learn_rate": self.config["learn_rate"],
            "reg_lambda": self.config["reg_lambda"],
            "GCN_layer": self.config["GCN_layer"],
            "delta": self.delta,
            "augmentation_direction": self.augmentation_direction,
            "unknown_taxonomy_policy": str(self.config["unknown_taxonomy_policy"]),
            "resume_checkpoint": str(self.config.get("resume_checkpoint", "none")),
            "sparsity_test": self.config["sparsity_test"],
            "dataset": str(self.config["dataset"]),
            "taxonomy_policy": str(self.config["taxonomy_policy"]),
            "direction_source": self.direction_source,
            "prototype_mode": self.prototype_mode,
            "prototype_update": self.prototype_update,
            "prototype_init_scope": self.prototype_init_scope,
            "epsilon_max": self.epsilon_max,
            "ssl_lambda": self.ssl_lambda,
            "temperature": self.temperature,
            "mu": float(self.config["mu"]),
            "warm_start_epochs": self.warm_start_epochs,
            "same_leaf_weight": self.same_leaf_weight,
            "use_leaf_aware_infonce": self.use_leaf_aware_infonce,
            "symmetric_info_nce": self.symmetric_info_nce,
            "isotropic_blend": self.isotropic_blend,
            "use_adaptive_epsilon": self.use_adaptive_epsilon,
            "use_user_ssl": self.use_user_ssl,
            "prototype_weighting": self.prototype_weighting,
            "prototype_leave_one_out": self.prototype_leave_one_out,
            "asymmetric_view_direction": self.asymmetric_view_direction,
        }
        if self.prototype_mode == "mixture":
            config["mixture_alpha"] = self.mixture_alpha
        if self.use_adaptive_epsilon:
            config["gamma_cold"] = self.gamma_cold
            config["gamma_warm"] = self.gamma_warm
            config["degree_near_cold_max"] = self.degree_near_cold_max
            config["degree_long_tail_max"] = self.degree_long_tail_max
        if self.use_user_ssl:
            config["epsilon_user"] = self.epsilon_user
            config["ssl_lambda_user"] = self.ssl_lambda_user
            config["temperature_user"] = self.temperature_user
        return config

    def validate_checkpoint_metadata(self, metadata):
        expected = self.checkpoint_metadata()
        for key in (
            "model",
            "dataset",
            "taxonomy_policy",
            "taxonomy_granularity",
            "taxonomy_hash",
            "taxonomy_protocol_version",
            "evaluation_protocol_hash",
            "split_hashes",
            "unknown_taxonomy_policy",
            "symmetric_info_nce",
            "prototype_mode",
            "mixture_alpha",
            "perturbation_structure",
            "direction_source",
            "use_adaptive_epsilon",
            "use_leaf_aware_infonce",
            "gamma_cold",
            "gamma_warm",
            "same_leaf_weight",
            "degree_near_cold_max",
            "degree_long_tail_max",
            "use_user_ssl",
            "epsilon_user",
            "ssl_lambda_user",
            "temperature_user",
            "isotropic_blend",
        ):
            if metadata.get(key) != expected[key]:
                raise ValueError("Checkpoint metadata mismatch: {}".format(key))


class Trainer:
    def __init__(self, args, config, dataset, device, logger):
        self.model = TaxProCLImproved(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    def train(self):
        return taxpro_trainer.train_taxprocl(
            self.model,
            self.args,
            self.config,
            self.dataset,
            self.device,
            self.logger,
        )
