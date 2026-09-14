"""Controlled Week-5 subset smoke/stability checks using real taxonomy artifacts.

These diagnostics are engineering evidence only and must not be reported as
paper results.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.utility_data.taxonomy import load_taxonomy
from utility.utility_function.taxonomy_contrastive import (
    PrototypeBank,
    create_item_views,
    item_level_info_nce,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "smoke_stability" / "week5_smoke.json",
    )
    return parser.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def controlled_subset(taxonomy, per_leaf=24):
    valid_ids = np.flatnonzero(taxonomy.valid_train_mask)
    by_leaf = {}
    for item in valid_ids:
        by_leaf.setdefault(int(taxonomy.item_to_leaf_id[item]), []).append(int(item))
    chosen_leaves = [leaf for leaf, items in sorted(by_leaf.items()) if len(items) >= per_leaf][:3]
    if len(chosen_leaves) < 2:
        raise ValueError("Controlled subset needs at least two populated leaves")
    selected = []
    for leaf in chosen_leaves:
        selected.extend(by_leaf[leaf][:per_leaf])
    original_leaves = taxonomy.item_to_leaf_id[selected]
    remap = {leaf: index for index, leaf in enumerate(chosen_leaves)}
    local_leaves = torch.tensor(
        [remap[int(leaf)] for leaf in original_leaves], dtype=torch.long
    )
    return selected, local_leaves, chosen_leaves


def run_scenario(dataset, policy, epochs, device):
    seed_everything(42)
    taxonomy = load_taxonomy(dataset, policy)
    selected, leaves_cpu, source_leaves = controlled_subset(taxonomy)
    leaves = leaves_cpu.to(device)
    embeddings = torch.nn.Parameter(
        torch.empty(len(selected), 32, device=device)
    )
    torch.nn.init.xavier_uniform_(embeddings)
    optimizer = torch.optim.Adam([embeddings], lr=0.003)
    bank = PrototypeBank(len(source_leaves), 32, mu=0.9).to(device)
    valid = torch.ones(len(selected), dtype=torch.bool, device=device)
    initialization = bank.initialize(embeddings.detach(), leaves, valid, epoch=1)
    losses, drift, direction_cosines = [], [], []
    augmentation_norm_max = 0.0
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        update = bank.ema_update(embeddings, torch.arange(len(selected), device=device), leaves, valid)
        view_a, view_b, view_evidence = create_item_views(
            embeddings,
            leaves,
            valid,
            bank.prototypes,
            epsilon_max=0.1,
            delta=1e-8,
            direction_mode="taxonomy",
        )
        loss = item_level_info_nce(view_a, view_b, temperature=0.2)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite contrastive loss")
        loss.backward()
        if bank.prototypes.grad is not None:
            raise AssertionError("Prototype buffer received gradients")
        losses.append(float(loss.detach().cpu()))
        drift.append(float(update["prototype_drift_mean"]))
        max_epsilon = max(
            float(view_evidence["epsilon_a"].max().detach().cpu()),
            float(view_evidence["epsilon_b"].max().detach().cpu()),
        )
        augmentation_norm_max = max(augmentation_norm_max, max_epsilon)
        if max_epsilon > 0.1000001:
            raise AssertionError("Augmentation exceeded epsilon_max")
        target_direction = bank.prototypes[leaves] - embeddings
        nonzero = target_direction.norm(dim=1) > 1e-8
        if bool(nonzero.any().item()):
            cosine = functional.cosine_similarity(
                view_evidence["direction"][nonzero],
                target_direction[nonzero],
                dim=1,
            )
            direction_cosines.extend(cosine.detach().cpu().tolist())
        optimizer.step()
    direction_cosine_min = min(direction_cosines) if direction_cosines else 1.0
    if direction_cosine_min <= 0.999:
        raise AssertionError("Taxonomy augmentation direction criterion failed")

    loss_explosion_limit = max(10.0, losses[0] * 10.0)
    loss_exploded = max(losses) > loss_explosion_limit
    if loss_exploded:
        raise AssertionError("Stability loss explosion criterion failed")

    for _ in range(100):
        bank.ema_update(
            embeddings,
            torch.arange(len(selected), device=device),
            leaves,
            valid,
        )
    ema_probe_finite = bool(torch.isfinite(bank.prototypes).all().item())
    if not ema_probe_finite:
        raise AssertionError("Prototype bank became non-finite after 100 EMA updates")
    normalized = functional.normalize(embeddings.detach(), dim=1)
    different = leaves[:, None] != leaves[None, :]
    cross_leaf = normalized @ normalized.transpose(0, 1)
    cross_values = cross_leaf[different]
    mean_cross_cosine = float(cross_values.mean().cpu())
    finite = all(
        math.isfinite(value)
        for value in losses
        + drift
        + bank.prototypes.detach().cpu().norm(dim=1).tolist()
    )
    if not finite or mean_cross_cosine > 0.9:
        raise AssertionError("Stability/collapse criterion failed")

    checkpoint = {
        "embedding": embeddings.detach().cpu(),
        "prototype_state": bank.state_dict(),
        "taxonomy_hash": taxonomy.taxonomy_hash,
        "dataset": dataset,
        "policy": policy,
    }
    restored = PrototypeBank(len(source_leaves), 32, mu=0.9)
    restored.load_state_dict(checkpoint["prototype_state"])
    checkpoint_round_trip = torch.equal(
        restored.prototypes, bank.prototypes.detach().cpu()
    )
    if not checkpoint_round_trip:
        raise AssertionError("Prototype checkpoint round-trip failed")
    return {
        "status": "PASS",
        "engineering_evidence_only": True,
        "dataset": dataset,
        "taxonomy_policy": policy,
        "seed": 42,
        "epochs": int(epochs),
        "subset_items": len(selected),
        "source_leaf_ids": source_leaves,
        "taxonomy_hash": taxonomy.taxonomy_hash,
        "prototype_initialization": initialization,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_max": max(losses),
        "loss_explosion_limit": loss_explosion_limit,
        "loss_exploded": loss_exploded,
        "prototype_drift_max": max(drift),
        "prototype_norms_finite": finite,
        "prototype_has_gradient": bank.prototypes.grad is not None,
        "ema_probe_batches": 100,
        "ema_probe_finite": ema_probe_finite,
        "augmentation_norm_max": augmentation_norm_max,
        "augmentation_direction_cosine_min": direction_cosine_min,
        "mean_cross_leaf_cosine": mean_cross_cosine,
        "checkpoint_round_trip": checkpoint_round_trip,
    }


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    scenarios = (
        ("yelp2018", "no_merge", 5),
        ("yelp2018", "merge_t10", 5),
        ("amazon-book", "no_merge", 5),
        ("yelp2018", "no_merge", 50),
    )
    rows = [
        run_scenario(dataset, policy, epochs, device)
        for dataset, policy, epochs in scenarios
    ]
    payload = {
        "schema_version": 1,
        "purpose": "Week-5 subset smoke and stability engineering evidence",
        "paper_result_eligible": False,
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "scenarios": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
