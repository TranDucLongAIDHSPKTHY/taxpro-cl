"""Mechanism diagnostics on existing checkpoints (docs/extended_evaluation_protocol.md), by
training-degree group and propagation layer. Descriptive: no inference is drawn from them.

Inference only, no training, and no test split is read: ranking measures use the
validation split, which excludes train positives. Input: the best-validation checkpoints
of V0-V3 on the five datasets, the families of FAMILIES in tools/experiments/run_extended.py.
Fixed minibatches run through the training forward pass, which draws two perturbed views.

The run fails if the realized perturbation does not follow the configured degree scaling:
between adaptive and fixed arms the ratio must stay within RATIO_TOLERANCE of gamma_cold for
near-cold items and of 1 for mid-tail and warm items, per layer.

Usage:
    python -m tools.analysis.mechanism_diagnostics
    python -m tools.analysis.mechanism_diagnostics --datasets musical-instruments --seeds 42
Output: results/extended/mechanism_diagnostics.json
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir  # noqa: E402
from tools.experiments.run_confirmatory import best_validation  # noqa: E402
from tools.experiments.run_extended import FAMILIES  # noqa: E402
from tools.ranking.inference import compute_batch_order_and_rank, load_model  # noqa: E402
from utility.utility_train.group_evaluator import load_targets  # noqa: E402

logger = logging.getLogger("mechanism_diagnostics")

SEEDS = (42, 0, 1)
ARMS = ("V0", "V1", "V2", "V3")
ADAPTIVE_PAIRS = (("V2", "V0"), ("V3", "V1"))
ARM_PAIRS = ADAPTIVE_PAIRS + (("V1", "V0"), ("V3", "V2"))
BINS = ("near_cold", "mid_tail", "warm")
GROUPS = ("near_cold", "long_tail", "warm", "overall")
BATCHES = 50
GENERATOR_SEED = 7
RATIO_TOLERANCE = 0.02
RANK_BATCH = 256
TOP = 20
RECALL_CUTOFFS = (20, 50, 100)
NDCG_CUTOFF = 100


def degree_masks(degrees):
    degrees = np.asarray(degrees, dtype=np.int64)
    return {
        "near_cold": (degrees >= 1) & (degrees <= 5),
        "mid_tail": (degrees >= 6) & (degrees <= 10),
        "warm": degrees > 10,
    }


def by_bin(values, masks):
    return {name: float(np.mean(values[masks[name]])) if masks[name].any() else None for name in BINS}


def by_bin_and_layer(values, masks):
    """values: [layers, items]. Each group gets its per-layer means and their mean."""
    result = {}
    for name in BINS:
        if not masks[name].any():
            result[name] = None
            continue
        per_layer = [float(np.mean(row[masks[name]])) for row in values]
        result[name] = {"per_layer": per_layer, "mean": float(np.mean(per_layer))}
    return result


def exposure(dataset, masks, valid_mask, batch_size):
    """Share of training interactions and of item-side InfoNCE anchors per group, over
    one simulated epoch: anchors are the distinct valid positives of each minibatch."""
    items = np.asarray(dataset.train_item, dtype=np.int64)
    interactions = {name: float(masks[name][items].mean()) for name in BINS}
    order = np.random.default_rng(GENERATOR_SEED).permutation(len(items))
    anchors = {name: 0 for name in BINS}
    for start in range(0, len(order), batch_size):
        batch = np.unique(items[order[start:start + batch_size]])
        batch = batch[valid_mask[batch]]
        for name in BINS:
            anchors[name] += int(masks[name][batch].sum())
    total = sum(anchors.values())
    return {"interaction_share": interactions,
            "anchor_share": {name: anchors[name] / total for name in BINS}}


class PerturbationProbe:
    """Records the item perturbation of each layer and the two final views, by wrapping
    the bound methods of one model instance while the training forward pass runs."""

    def __init__(self, model):
        self.model = model
        self.layers = int(model.config["GCN_layer"])
        self.num_users = model.dataset.num_users
        shape = (self.layers, model.dataset.num_items)
        device = model.item_embedding.weight.device
        self.norm = torch.zeros(shape, dtype=torch.float64, device=device)
        self.ratio = torch.zeros(shape, dtype=torch.float64, device=device)
        self.cosine = torch.zeros(shape[1], dtype=torch.float64, device=device)
        self.calls = 0
        self.pairs = 0
        self.first_view = None

    def __enter__(self):
        self.perturb = self.model._perturb_items
        self.aggregate = self.model.aggregate
        self.model._perturb_items = self.wrapped_perturb
        self.model.aggregate = self.wrapped_aggregate
        return self

    def __exit__(self, *_):
        del self.model._perturb_items
        del self.model.aggregate

    def wrapped_perturb(self, all_embedding, gcn_layers, view_id=None):
        output = self.perturb(all_embedding, gcn_layers, view_id=view_id)
        before = all_embedding[self.num_users:].detach()
        realized = (output[self.num_users:].detach() - before).norm(dim=1).double()
        layer = self.calls % self.layers
        self.norm[layer] += realized
        self.ratio[layer] += realized / before.norm(dim=1).double().clamp_min(1e-12)
        self.calls += 1
        return output

    def wrapped_aggregate(self, perturbed=False, view_id=None):
        users, items = self.aggregate(perturbed=perturbed, view_id=view_id)
        if perturbed and view_id == 0:
            self.first_view = items.detach()
        elif perturbed and view_id == 1:
            self.cosine += torch.nn.functional.cosine_similarity(self.first_view, items.detach(), dim=1).double()
            self.pairs += 1
        return users, items

    def results(self):
        passes = self.calls / self.layers
        return ((self.norm / passes).cpu().numpy(), (self.ratio / passes).cpu().numpy(),
                (self.cosine / self.pairs).cpu().numpy())


def forward_measures(model, dataset, device, batches):
    """Runs fixed minibatches through the training forward pass. The gradient is that of
    the item-side InfoNCE alone, unweighted, with respect to the layer-0 item table. The
    prototype bank is restored after each batch, so every batch sees the checkpoint's bank."""
    random.seed(GENERATOR_SEED)
    np.random.seed(GENERATOR_SEED)
    torch.manual_seed(GENERATOR_SEED)
    batch_size = int(model.config["batch_size"])
    sample = torch.as_tensor(dataset.sample_data_to_train_all(), dtype=torch.long)
    order = torch.randperm(sample.shape[0], generator=torch.Generator().manual_seed(GENERATOR_SEED))
    sample = sample[order[: batches * batch_size]].to(device)
    bank = {key: value.clone() for key, value in model.prototype_bank.state_dict().items()}
    model.ssl_lambda = 1.0
    model.ssl_lambda_user = 0.0
    epoch = model.warm_start_epochs + 1
    gradient = torch.zeros(dataset.num_items, dtype=torch.float64, device=device)
    used = 0
    with PerturbationProbe(model) as probe:
        for start in range(0, sample.shape[0], batch_size):
            batch = sample[start:start + batch_size]
            model.zero_grad(set_to_none=True)
            losses = model(batch[:, 0], batch[:, 1], batch[:, 2], epoch)
            losses[2].backward()
            gradient += model.item_embedding.weight.grad.norm(dim=1).double()
            used += 1
            model.prototype_bank.load_state_dict(bank)
    model.zero_grad(set_to_none=True)
    norm, ratio, cosine = probe.results()
    return norm, ratio, cosine, (gradient / used).cpu().numpy()


def ranking_on_validation(model, dataset, targets, device):
    """Top-20 lists of validation users and rank-based measures per group. Candidates
    are the full catalog minus each user's train positives."""
    users = sorted(targets["overall"])
    top = np.zeros((len(users), TOP), dtype=np.int32)
    sums = {group: {"users": 0, "percentile": 0.0, "ndcg": 0.0,
                    **{f"recall_{k}": 0.0 for k in RECALL_CUTOFFS}} for group in GROUPS}
    discounts = 1.0 / np.log2(np.arange(2, NDCG_CUTOFF + 2))
    for start in range(0, len(users), RANK_BATCH):
        batch = users[start:start + RANK_BATCH]
        order, rank = compute_batch_order_and_rank(model, dataset, device, batch, split="validation")
        top[start:start + len(batch)] = order[:, :TOP].cpu().numpy()
        rank = rank.cpu().numpy()
        for row, user in enumerate(batch):
            candidates = dataset.num_items - dataset.user_item_net[user].nnz
            for group in GROUPS:
                items = targets[group].get(user)
                if not items:
                    continue
                ranks = rank[row, np.fromiter(items, dtype=np.int64)]
                slot = sums[group]
                slot["users"] += 1
                slot["percentile"] += float(np.mean(1.0 - (ranks - 1) / candidates))
                for cutoff in RECALL_CUTOFFS:
                    slot[f"recall_{cutoff}"] += float(np.mean(ranks <= cutoff))
                hits = ranks[ranks <= NDCG_CUTOFF]
                ideal = discounts[: min(NDCG_CUTOFF, len(ranks))].sum()
                slot["ndcg"] += float((1.0 / np.log2(hits + 1)).sum() / ideal)
    measures = {}
    for group, slot in sums.items():
        count = slot.pop("users")
        measures[group] = {"users": count, **{key: value / count for key, value in slot.items()}}
    return top, measures, users


def measure_run(run_dir, masks, targets, device, batches):
    model, dataset, _, _ = load_model(run_dir, device)
    try:
        valid = model.valid_train_mask.cpu().numpy()
        valid_masks = {name: masks[name] & valid for name in BINS}
        norm, ratio, cosine, gradient = forward_measures(model, dataset, device, batches)
        total = float(gradient[valid].sum())
        result = {"realized_perturbation_norm": by_bin_and_layer(norm, valid_masks),
                  "perturbation_to_embedding_ratio": by_bin_and_layer(ratio, valid_masks),
                  "view_cosine": by_bin(cosine, valid_masks),
                  "item_contrastive_gradient_norm": by_bin(gradient, valid_masks),
                  "item_contrastive_gradient_share":
                      {name: float(gradient[valid_masks[name]].sum()) / total for name in BINS}}
        with torch.no_grad():
            _, clean = model.aggregate(perturbed=False)
        top, measures, users = ranking_on_validation(model, dataset, targets, device)
        stored = best_validation(Path(run_dir))
        for group in GROUPS:
            measures[group]["stored_validation_recall_20"] = stored[group]
        result["validation_measures"] = measures
        extras = {"clean": clean.detach().cpu().numpy(), "top": top, "users": users, "valid": valid,
                  "batch_size": int(model.config["batch_size"]), "gamma_cold": float(model.gamma_cold)}
        return result, extras, dataset
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def jaccard(left, right, chunk=4096):
    values = np.zeros(left.shape[0], dtype=np.float64)
    for start in range(0, left.shape[0], chunk):
        a, b = left[start:start + chunk], right[start:start + chunk]
        shared = (a[:, :, None] == b[:, None, :]).sum(axis=(1, 2))
        values[start:start + chunk] = shared / (2 * TOP - shared)
    return values


def overlap_by_group(left, right, users, targets):
    values = jaccard(left, right)
    position = {user: index for index, user in enumerate(users)}
    return {group: float(values[[position[user] for user in targets[group] if user in position]].mean())
            for group in GROUPS}


def embedding_shift(reference, other, masks):
    """||other - reference|| / ||reference|| per group."""
    denominator = np.linalg.norm(reference, axis=1)
    values = np.divide(np.linalg.norm(other - reference, axis=1), denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    return by_bin(values, masks)


def embedding_cosine(reference, other, masks):
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(other, axis=1)
    values = np.divide((reference * other).sum(axis=1), denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    return by_bin(values, masks)


def scaling_check(runs, gamma_cold, seeds):
    """Ratio of the realized perturbation norm between adaptive and fixed arms, per layer."""
    expected = {"near_cold": gamma_cold, "mid_tail": 1.0, "warm": 1.0}
    rows = []
    for adaptive, fixed in ADAPTIVE_PAIRS:
        for seed in seeds:
            top = runs[adaptive][str(seed)]["realized_perturbation_norm"]
            bottom = runs[fixed][str(seed)]["realized_perturbation_norm"]
            for name, target in expected.items():
                if top[name] is None or bottom[name] is None:
                    continue
                for layer, (a, b) in enumerate(zip(top[name]["per_layer"], bottom[name]["per_layer"])):
                    ratio = a / b
                    rows.append({"pair": f"{adaptive}/{fixed}", "seed": seed, "group": name, "layer": layer,
                                 "ratio": ratio, "expected": target,
                                 "within_tolerance": abs(ratio / target - 1.0) <= RATIO_TOLERANCE})
    return rows


def analyse_dataset(dataset_name, seeds, device, batches):
    degrees = json.loads((evaluation_protocol_dir(dataset_name) / "item_degrees_train.json")
                         .read_text(encoding="utf-8"))["degrees"]
    masks = degree_masks(degrees)
    targets = load_targets(evaluation_protocol_dir(dataset_name), "validation")
    runs, cache, block = {}, {}, None
    for arm in ARMS:
        for seed in seeds:
            run_dir = ROOT / FAMILIES[dataset_name][arm] / f"seed{seed}"
            started = time.time()
            result, extras, dataset = measure_run(run_dir, masks, targets, device, batches)
            runs.setdefault(arm, {})[str(seed)] = result
            cache[arm, seed] = extras
            logger.info("%s %s seed%s done in %.0fs", dataset_name, arm, seed, time.time() - started)
            if block is None:
                block = {"items_per_bin": {name: int(masks[name].sum()) for name in BINS},
                         "gamma_cold": extras["gamma_cold"],
                         "exposure": exposure(dataset, masks, extras["valid"], extras["batch_size"])}
    pairs = {}
    for first, second in ARM_PAIRS:
        rows = {}
        for seed in seeds:
            a, b = cache[first, seed], cache[second, seed]
            rows[str(seed)] = {"embedding_shift": embedding_shift(b["clean"], a["clean"], masks),
                               "embedding_cosine": embedding_cosine(b["clean"], a["clean"], masks),
                               "top20_overlap": overlap_by_group(a["top"], b["top"], a["users"], targets)}
        pairs[f"{first}-{second}"] = rows
    seed_reference = {}
    for arm in ARMS:
        rows = {}
        for index, left in enumerate(seeds):
            for right in seeds[index + 1:]:
                a, b = cache[arm, left], cache[arm, right]
                rows[f"{left}-{right}"] = overlap_by_group(a["top"], b["top"], a["users"], targets)
        seed_reference[arm] = rows
    block.update(runs=runs, arm_pairs=pairs, seed_pair_top20_overlap=seed_reference,
                 ranking_change_ratio=ranking_change_ratio(pairs, seed_reference),
                 scaling_check=scaling_check(runs, block["gamma_cold"], seeds))
    return block


def ranking_change_ratio(pairs, seed_reference):
    """r = (1 - J(arm pair)) / (1 - J(seed pair)), per group, as seed means."""
    ratio = {}
    reference = {group: statistics.fmean(1.0 - row[group] for rows in seed_reference.values()
                                         for row in rows.values()) for group in GROUPS} \
        if any(seed_reference.values()) else None
    for name, rows in pairs.items():
        change = {group: statistics.fmean(1.0 - row["top20_overlap"][group] for row in rows.values())
                  for group in GROUPS}
        ratio[name] = {"delta_arm": change, "delta_seed": reference,
                       "r": None if reference is None else
                       {group: change[group] / reference[group] if reference[group] else None
                        for group in GROUPS}}
    return ratio


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--batches", type=int, default=BATCHES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "extended" / "mechanism_diagnostics.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    document = {
        "protocol": "docs/extended_evaluation_protocol.md",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True, check=True).stdout.strip(),
        "batches": args.batches, "generator_seed": GENERATOR_SEED, "seeds": args.seeds, "datasets": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for dataset_name in args.datasets:
        block = analyse_dataset(dataset_name, tuple(args.seeds), device, args.batches)
        document["datasets"][dataset_name] = block
        args.output.write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))
        failures += [{"dataset": dataset_name, **row} for row in block["scaling_check"]
                     if not row["within_tolerance"]]
        logger.info("%s written to %s", dataset_name, args.output)
    if failures:
        logger.error("%d scaling checks failed; stop before using these results", len(failures))
        for row in failures[:20]:
            logger.error("%s", row)
        return 1
    logger.info("All scaling checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
