"""Rigorous check for the rescue variant vs A2-V0 / A2-V3 (prompted by
seeing all-positive-but-small group-level deltas): per-user
Recall@20 bootstrap CI, same methodology as
tools/analysis/seed_matched_bootstrap.py (the paper's own primary
significance evidence for Table 8/RQ1), applied to seed-matched checkpoint
pairs (42-42, 0-0, 1-1) instead of TaxPro-CL-vs-baseline. Each user's diff
is averaged across the 3 seed pairs it appears in BEFORE bootstrapping, so
each user contributes exactly one value to the resampled population (the
previous version appended each user's per-seed diff directly into the
pooled list inside the seed loop, pseudo-replicating every user up to 3x
and understating the interval width -- confirmed via the exact 3x #Users
multiplier in the previously published tables).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir
from utility.utility_train.group_evaluator import load_targets
from tests.Recommendation_system.inference import load_model, compute_batch_order_and_rank

DATASET = "amazon-book"
SEEDS = ["42", "0", "1"]
GROUPS = ["near_cold", "long_tail", "overall", "warm"]
K = 20
BATCH_SIZE = 256
N_BOOT = 5000


def per_user_recall(model, dataset, device, targets_by_group):
    all_users = sorted(set().union(*[set(t.keys()) for t in targets_by_group.values()]))
    per_user = {g: {} for g in GROUPS}
    for start in range(0, len(all_users), BATCH_SIZE):
        batch = all_users[start:start + BATCH_SIZE]
        order, _rank = compute_batch_order_and_rank(model, dataset, device, batch, split="test")
        topk = set()
        order_np = order[:, :K].cpu().numpy()
        for row_idx, user in enumerate(batch):
            topk_items = set(order_np[row_idx].tolist())
            for g in GROUPS:
                positives = targets_by_group[g].get(user)
                if not positives:
                    continue
                hits = len(topk_items & set(positives))
                per_user[g][user] = hits / len(positives)
    return per_user


def bootstrap(diffs, n_boot, rng):
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    observed = float(arr.mean())
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = arr[idx].mean()
    boot.sort()
    lo = float(boot[int(0.025 * n_boot)])
    hi = float(boot[int(0.975 * n_boot) - 1])
    return {
        "n_users": n,
        "mean_diff": observed,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "excludes_zero": bool((lo > 0) or (hi < 0)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.WARNING)
    protocol_dir = evaluation_protocol_dir(DATASET)
    targets_by_group = load_targets(protocol_dir, "test")

    per_user_v0 = {g: {} for g in GROUPS}
    per_user_v3 = {g: {} for g in GROUPS}

    for seed in SEEDS:
        print(f"=== seed {seed} ===")
        rescue_dir = ROOT / "log/p0/taxprocl/amazon-book/gvhd-A3-rescue" / f"seed{seed}"
        v0_dir = ROOT / "log/p0/taxprocl/amazon-book/A2-V0" / f"seed{seed}"
        v3_dir = ROOT / "log/p0/taxprocl/amazon-book/A2-V3" / f"seed{seed}"

        rescue_model, rescue_ds, _c, _n = load_model(rescue_dir, device)
        pu_rescue = per_user_recall(rescue_model, rescue_ds, device, targets_by_group)
        del rescue_model

        v0_model, v0_ds, _c, _n = load_model(v0_dir, device)
        pu_v0 = per_user_recall(v0_model, v0_ds, device, targets_by_group)
        del v0_model

        v3_model, v3_ds, _c, _n = load_model(v3_dir, device)
        pu_v3 = per_user_recall(v3_model, v3_ds, device, targets_by_group)
        del v3_model

        for g in GROUPS:
            common_v0 = set(pu_rescue[g]) & set(pu_v0[g])
            for u in common_v0:
                per_user_v0[g].setdefault(u, []).append(pu_rescue[g][u] - pu_v0[g][u])
            common_v3 = set(pu_rescue[g]) & set(pu_v3[g])
            for u in common_v3:
                per_user_v3[g].setdefault(u, []).append(pu_rescue[g][u] - pu_v3[g][u])

    # Average each user's diff across the seed pairs they appear in -- one
    # value per unique user -- before bootstrapping.
    pooled_diffs_v0 = {g: [float(np.mean(vals)) for vals in per_user_v0[g].values()] for g in GROUPS}
    pooled_diffs_v3 = {g: [float(np.mean(vals)) for vals in per_user_v3[g].values()] for g in GROUPS}

    rng = np.random.default_rng(42)
    results = {"vs_V0": {}, "vs_V3": {}}
    print("\n=== Pooled bootstrap (3 seed-pairs pooled), rescue vs V0 ===")
    for g in GROUPS:
        stat = bootstrap(pooled_diffs_v0[g], N_BOOT, rng)
        results["vs_V0"][g] = stat
        print(f"{g}: n={stat['n_users']} mean_diff={stat['mean_diff']:+.6f} "
              f"95% CI=[{stat['ci95_lo']:+.6f}, {stat['ci95_hi']:+.6f}] excludes_zero={stat['excludes_zero']}")

    print("\n=== Pooled bootstrap (3 seed-pairs pooled), rescue vs V3 ===")
    for g in GROUPS:
        stat = bootstrap(pooled_diffs_v3[g], N_BOOT, rng)
        results["vs_V3"][g] = stat
        print(f"{g}: n={stat['n_users']} mean_diff={stat['mean_diff']:+.6f} "
              f"95% CI=[{stat['ci95_lo']:+.6f}, {stat['ci95_hi']:+.6f}] excludes_zero={stat['excludes_zero']}")

    out = ROOT / "results" / "rescue_vs_variants_bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved to", out)


if __name__ == "__main__":
    main()
