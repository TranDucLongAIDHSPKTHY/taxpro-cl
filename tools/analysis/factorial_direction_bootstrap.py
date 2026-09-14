"""Per-user bootstrap CI for the V0-V3 direction-by-magnitude factorial
(manuscript Section 5.4 "Direction-by-magnitude factorial ablation (RQ5)",
Table 11).

V0/V1/V2/V3 hold every hyperparameter fixed within a dataset except
perturbation direction (random vs. taxonomy) and epsilon-adaptivity
(fixed vs. adaptive):

    V0 = random direction,   fixed epsilon
    V1 = taxonomy direction, fixed epsilon
    V2 = random direction,   adaptive epsilon
    V3 = taxonomy direction, adaptive epsilon (= the main configuration)

This is a full 2x2 factorial (direction x epsilon-adaptivity), so it
supports isolating either factor from the same four trained variants. It
computes, for each dataset, the per-user paired Recall@20 difference for
all four marginal comparisons, pooled across the three trained seeds (42,
0, 1) before bootstrap-resampling over users -- the same procedure as
tools/analysis/seed_matched_bootstrap.py's pooled CI, applied here to
internal variant pairs instead of TaxPro-CL vs. a baseline:

    V1-V0: direction's effect at fixed epsilon
    V3-V2: direction's effect at adaptive epsilon
    V2-V0: epsilon-adaptivity's effect at random direction
    V3-V1: epsilon-adaptivity's effect at taxonomy direction

Requires the V0/V1/V2 runs to already exist (produced by the A2 factorial
commands documented in the paper's Section 5.4 methodology); V3
reuses each dataset's main-configuration checkpoint (same one
checkpoint_selection.select_checkpoint("TaxPro-CL", dataset) resolves to
for Yelp2018/Musical-Instruments/Arts-Crafts-and-Sewing, and the A2-V3
factorial run for Amazon-Book/Yelp2018). Inference-only, no retraining.

Usage:
    python -m tools.analysis.factorial_direction_bootstrap
    python -m tools.analysis.factorial_direction_bootstrap --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import RESULT_DIR, evaluation_protocol_dir
from utility.utility_train.group_evaluator import load_targets
from tests.Recommendation_system.inference import load_model, compute_batch_order_and_rank

SEEDS = ["42", "0", "1"]
GROUPS = ["near_cold", "long_tail", "overall", "warm"]
K = 20
BATCH_SIZE = 256
N_BOOT = 5000

# Run-directory layout, relative to the project root. V3 for Musical-
# Instruments/Arts-Crafts-and-Sewing reuses each dataset's locked main
# checkpoint family instead of a separate gvhd-taxctrl-V3 run, since it is
# bit-for-bit the same configuration (taxonomy direction + adaptive epsilon).
DATASET_DIRS = {
    "amazon-book": {
        "V0": "log/p0/taxprocl/amazon-book/A2-V0",
        "V1": "log/p0/taxprocl/amazon-book/A2-V1",
        "V2": "log/p0/taxprocl/amazon-book/A2-V2",
        "V3": "log/p0/taxprocl/amazon-book/A2-V3",
    },
    "yelp2018": {
        "V0": "log/p0/taxprocl/yelp2018/A2-V0",
        "V1": "log/p0/taxprocl/yelp2018/A2-V1",
        "V2": "log/p0/taxprocl/yelp2018/A2-V2",
        "V3": "log/p0/taxprocl/yelp2018/A2-V3",
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
    return {
        "n_users": n,
        "mean_diff": observed,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "excludes_zero": bool((lo > 0) or (hi < 0)),
    }


def pool_diffs_across_seeds(pooled_by_user):
    """Collapse {user_id: [diff_seed0, diff_seed1, ...]} to one value per
    user (the mean across the seeds that user appeared in), producing
    exactly one bootstrap-eligible observation per unique user regardless
    of how many seeds contributed to it.

    Worked example: user 1 has
    per-seed diffs [1, 1, 1], user 2 has [-1, -1, -1]. Pooling naively by
    appending every (user, seed) pair into one flat list -- the bug this
    function fixes -- would treat that as 6 independent observations,
    3 of value +1 and 3 of value -1. Pooling correctly per user first
    collapses it to exactly 2 observations, +1 and -1 (see
    tests/test_factorial_direction_bootstrap.py).
    """
    return [float(np.mean(diffs)) for diffs in pooled_by_user.values()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_DIRS), choices=list(DATASET_DIRS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "factorial_direction_bootstrap.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)
    logging.basicConfig(level=logging.WARNING)
    rng = np.random.default_rng(42)

    all_results = {}
    for dataset in args.datasets:
        dirs = DATASET_DIRS[dataset]
        print(f"\n########## {dataset} ########## elapsed={time.time()-t0:.0f}s", flush=True)
        protocol_dir = evaluation_protocol_dir(dataset)
        targets_by_group = load_targets(protocol_dir, "test")

        # Per-user, per-seed diffs are collected here first (each user keyed
        # once, with one list entry appended per seed pair it appears in);
        # only AFTER the seed loop are they averaged into one value per user
        # (see below). Appending directly into a flat pooled list inside the
        # seed loop -- the previous approach -- pseudo-replicates each user
        # up to 3x (once per seed) and understates bootstrap uncertainty
        # (confirmed via the exact 3x #Users multiplier in the published
        # tables).
        pooled_by_user = {
            "V1_vs_V0": {g: {} for g in GROUPS},
            "V3_vs_V2": {g: {} for g in GROUPS},
            "V2_vs_V0": {g: {} for g in GROUPS},
            "V3_vs_V1": {g: {} for g in GROUPS},
        }

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
                for u in set(pu1[g]) & set(pu0[g]):
                    pooled_by_user["V1_vs_V0"][g].setdefault(u, []).append(pu1[g][u] - pu0[g][u])
                for u in set(pu3[g]) & set(pu2[g]):
                    pooled_by_user["V3_vs_V2"][g].setdefault(u, []).append(pu3[g][u] - pu2[g][u])
                for u in set(pu2[g]) & set(pu0[g]):
                    pooled_by_user["V2_vs_V0"][g].setdefault(u, []).append(pu2[g][u] - pu0[g][u])
                for u in set(pu3[g]) & set(pu1[g]):
                    pooled_by_user["V3_vs_V1"][g].setdefault(u, []).append(pu3[g][u] - pu1[g][u])

        # Average each user's diff across the seed pairs they appear in --
        # exactly one value per unique user, matching seed_matched_bootstrap.py's
        # (already-correct) methodology, before bootstrap resampling.
        pooled = {
            key: {g: pool_diffs_across_seeds(pooled_by_user[key][g]) for g in GROUPS}
            for key in pooled_by_user
        }

        labels = {
            "V1_vs_V0": "direction's effect at fixed epsilon",
            "V3_vs_V2": "direction's effect at adaptive epsilon",
            "V2_vs_V0": "epsilon-adaptivity's effect at random direction",
            "V3_vs_V1": "epsilon-adaptivity's effect at taxonomy direction",
        }
        ds_results = {}
        for key, label in labels.items():
            ds_results[key] = {}
            print(f"-- {dataset}: {key.replace('_vs_', ' vs ')} ({label}) --")
            for g in GROUPS:
                stat = bootstrap(pooled[key][g], args.n_boot, rng)
                ds_results[key][g] = stat
                print(f"{g}: n={stat['n_users']} mean_diff={stat['mean_diff']:+.6f} "
                      f"95% CI=[{stat['ci95_lo']:+.6f}, {stat['ci95_hi']:+.6f}] excludes_zero={stat['excludes_zero']}")

        all_results[dataset] = ds_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("\nSaved to", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
