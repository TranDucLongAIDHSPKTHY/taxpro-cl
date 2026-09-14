"""Interaction-effect bootstrap for the V0-V3 direction-by-epsilon-adaptivity
factorial (GVHD review 2026-09-12, item A5, point 3): computes, per dataset
and group, the interaction term (V3-V2)-(V1-V0) from the SAME per-user,
per-seed data used for the two conditional (marginal) effects -- not two
independently-resampled bootstraps -- so the interaction's uncertainty
correctly reflects the shared dependency between the two conditional
effects (both are computed from the same four checkpoints and the same
users).

Terminology, per GVHD's suggestion: V1-V0 and V3-V2 are "direction's
conditional effect" (at fixed epsilon and at adaptive epsilon,
respectively); V2-V0 and V3-V1 are "epsilon-adaptivity's conditional
effect" (at random direction and at taxonomy direction, respectively);
the interaction (V3-V2)-(V1-V0) [equivalently (V3-V1)-(V2-V0)] tests
whether direction's effect depends on the epsilon-adaptivity setting.

Reuses the corrected Yelp2018 checkpoints (temperature_user=0.15) and the
original checkpoints for Amazon-Book/Musical-Instruments/Arts-Crafts-and-
Sewing. Inference-only, no retraining.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir
from utility.utility_train.group_evaluator import load_targets
from tests.Recommendation_system.inference import load_model, compute_batch_order_and_rank

SEEDS = ["42", "0", "1"]
GROUPS = ["near_cold", "long_tail", "overall", "warm"]
K = 20
BATCH_SIZE = 256
N_BOOT = 5000

DATASET_DIRS = {
    "amazon-book": {
        "V0": "log/p0/taxprocl/amazon-book/A2-V0",
        "V1": "log/p0/taxprocl/amazon-book/A2-V1",
        "V2": "log/p0/taxprocl/amazon-book/A2-V2",
        "V3": "log/p0/taxprocl/amazon-book/A2-V3",
    },
    "yelp2018": {
        "V0": "log/p0/taxprocl/yelp2018/A2-V0-tempuser0.15",
        "V1": "log/p0/taxprocl/yelp2018/A2-V1-tempuser0.15",
        "V2": "log/p0/taxprocl/yelp2018/A2-V2-tempuser0.15",
        "V3": "log/p0/taxprocl/yelp2018/A2-V3-tempuser0.15",
    },
    "musical-instruments": {
        "V0": "log/p0/taxprocl/musical-instruments/gvhd-taxctrl-V0",
        "V1": "log/p0/taxprocl/musical-instruments/gvhd-taxctrl-V1",
        "V2": "log/p0/taxprocl/musical-instruments/gvhd-taxctrl-V2",
        "V3": "log/p0/taxprocl/musical-instruments/taxpro-cl-FINAL-no_merge-temp0.15-gammacold5.0",
    },
    "arts-crafts-and-sewing": {
        "V0": "log/p0/taxprocl/arts-crafts-and-sewing/gvhd-taxctrl-V0",
        "V1": "log/p0/taxprocl/arts-crafts-and-sewing/gvhd-taxctrl-V1",
        "V2": "log/p0/taxprocl/arts-crafts-and-sewing/gvhd-taxctrl-V2",
        "V3": "log/p0/taxprocl/arts-crafts-and-sewing/taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0",
    },
}


def per_user_recall(model, dataset, device, targets_by_group):
    all_users = sorted(set().union(*[set(t.keys()) for t in targets_by_group.values()]))
    per_user = {g: {} for g in GROUPS}
    for start in range(0, len(all_users), BATCH_SIZE):
        batch = all_users[start:start + BATCH_SIZE]
        order, _rank = compute_batch_order_and_rank(model, dataset, device, batch, split="test")
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
    return {"n_users": n, "mean_diff": observed, "ci95_lo": lo, "ci95_hi": hi,
            "excludes_zero": bool((lo > 0) or (hi < 0))}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    rng = np.random.default_rng(42)
    t0 = time.time()

    all_results = {}
    for dataset, dirs in DATASET_DIRS.items():
        print(f"\n########## {dataset} ########## elapsed={time.time()-t0:.0f}s", flush=True)
        protocol_dir = evaluation_protocol_dir(dataset)
        targets_by_group = load_targets(protocol_dir, "test")

        # per-user per-seed: interaction[user] = (v3-v2) - (v1-v0), computed
        # from the SAME seed's four checkpoints for that user (preserves
        # the shared dependency GVHD's A5 point 3 asked for), appended once
        # per seed the user appears in, then averaged per user across seeds
        # exactly like every other bootstrap in this paper.
        pooled_by_user = {g: {} for g in GROUPS}
        cond_direction_fixed = {g: {} for g in GROUPS}   # V1-V0
        cond_direction_adaptive = {g: {} for g in GROUPS}  # V3-V2

        for seed in SEEDS:
            print(f"=== seed {seed} === elapsed={time.time()-t0:.0f}s", flush=True)
            m0, ds0, _c, _n = load_model(ROOT / dirs["V0"] / f"seed{seed}", device)
            pu0 = per_user_recall(m0, ds0, device, targets_by_group)
            del m0
            m1, ds1, _c, _n = load_model(ROOT / dirs["V1"] / f"seed{seed}", device)
            pu1 = per_user_recall(m1, ds1, device, targets_by_group)
            del m1
            m2, ds2, _c, _n = load_model(ROOT / dirs["V2"] / f"seed{seed}", device)
            pu2 = per_user_recall(m2, ds2, device, targets_by_group)
            del m2
            m3, ds3, _c, _n = load_model(ROOT / dirs["V3"] / f"seed{seed}", device)
            pu3 = per_user_recall(m3, ds3, device, targets_by_group)
            del m3

            for g in GROUPS:
                common = set(pu0[g]) & set(pu1[g]) & set(pu2[g]) & set(pu3[g])
                for u in common:
                    d_fixed = pu1[g][u] - pu0[g][u]
                    d_adaptive = pu3[g][u] - pu2[g][u]
                    interaction = d_adaptive - d_fixed
                    pooled_by_user[g].setdefault(u, []).append(interaction)
                    cond_direction_fixed[g].setdefault(u, []).append(d_fixed)
                    cond_direction_adaptive[g].setdefault(u, []).append(d_adaptive)

        ds_results = {}
        for g in GROUPS:
            interaction_vals = [float(np.mean(v)) for v in pooled_by_user[g].values()]
            fixed_vals = [float(np.mean(v)) for v in cond_direction_fixed[g].values()]
            adaptive_vals = [float(np.mean(v)) for v in cond_direction_adaptive[g].values()]
            stat_int = bootstrap(interaction_vals, N_BOOT, rng)
            stat_fixed = bootstrap(fixed_vals, N_BOOT, rng)
            stat_adaptive = bootstrap(adaptive_vals, N_BOOT, rng)
            ds_results[g] = {
                "direction_conditional_effect_fixed_epsilon_V1_minus_V0": stat_fixed,
                "direction_conditional_effect_adaptive_epsilon_V3_minus_V2": stat_adaptive,
                "interaction_V3V2_minus_V1V0": stat_int,
            }
            print(f"-- {dataset} {g}: interaction=(V3-V2)-(V1-V0) mean={stat_int['mean_diff']:+.6f} "
                  f"CI=[{stat_int['ci95_lo']:+.6f},{stat_int['ci95_hi']:+.6f}] excl0={stat_int['excludes_zero']}")

        all_results[dataset] = ds_results

    out_path = ROOT / "results" / "factorial_interaction_bootstrap.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("\nSaved to", out_path)


if __name__ == "__main__":
    main()
