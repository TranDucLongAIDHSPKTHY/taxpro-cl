"""B1: NDCG@20 for the RQ5 direction-by-magnitude
factorial (V0-V3), extending tools/analysis/factorial_direction_bootstrap.py
(which reports Recall@20 only, Table 11/S16) to NDCG@20 -- main paper
Section 5.4 states its factorial evidence is Recall@20-only; Online
Resource 1 Section S21 reports NDCG@20 for the six main-comparison methods
but not for the internal V0-V3 factorial checkpoints. This script closes
that specific gap: same checkpoints, same per-user pooling-across-seeds
procedure, same bootstrap machinery as factorial_direction_bootstrap.py,
computing NDCG@20 (main paper Eq. for NDCG@K) instead of Recall@20.

Inference only; no retraining; no new checkpoints created.
"""

from __future__ import annotations

import argparse
import json
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
from tools.analysis.factorial_direction_bootstrap import DATASET_DIRS, SEEDS, K, BATCH_SIZE

GROUPS = ["near_cold", "long_tail"]
N_BOOT = 5000

COMPARISONS = [
    ("V1", "V0", "direction_fixed_epsilon"),
    ("V3", "V2", "direction_adaptive_epsilon"),
]


def _dcg_and_idcg(topk_items, positives, k):
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(positives))))
    if idcg == 0.0:
        return None
    dcg = 0.0
    for rank_idx, item in enumerate(topk_items):
        if item in positives:
            dcg += 1.0 / np.log2(rank_idx + 2)
    return dcg / idcg


def per_user_ndcg(model, dataset, device, targets_by_group):
    all_users = sorted(set().union(*[set(t.keys()) for t in targets_by_group.values()]))
    per_user = {g: {} for g in GROUPS}
    for start in range(0, len(all_users), BATCH_SIZE):
        batch = all_users[start:start + BATCH_SIZE]
        order, _rank = compute_batch_order_and_rank(model, dataset, device, batch, split="test")
        order_np = order[:, :K].cpu().numpy()
        for row_idx, user in enumerate(batch):
            topk_items = order_np[row_idx].tolist()
            for g in GROUPS:
                positives = targets_by_group[g].get(user)
                if not positives:
                    continue
                ndcg = _dcg_and_idcg(topk_items, set(positives), K)
                if ndcg is not None:
                    per_user[g][user] = ndcg
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
    return [float(np.mean(diffs)) for diffs in pooled_by_user.values()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_DIRS), choices=list(DATASET_DIRS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "b1_factorial_ndcg.json")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    results = []

    for dataset_name in args.datasets:
        t0 = time.time()
        protocol_dir = evaluation_protocol_dir(dataset_name)
        targets = load_targets(protocol_dir, "test")
        targets_by_group = {g: targets[g] for g in GROUPS}

        per_variant_per_seed_ndcg = {}
        for variant, rel_dir in DATASET_DIRS[dataset_name].items():
            per_seed = []
            for seed in SEEDS:
                run_dir = ROOT / rel_dir / "seed{}".format(seed)
                if not (run_dir / "run_manifest.json").is_file():
                    print("SKIP {}/{} seed={}: missing checkpoint at {}".format(dataset_name, variant, seed, run_dir), file=sys.stderr)
                    continue
                model, ds, _cfg, _name = load_model(run_dir, device)
                per_user = per_user_ndcg(model, ds, device, targets_by_group)
                per_seed.append(per_user)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            per_variant_per_seed_ndcg[variant] = per_seed
            print("[{}] {}: scored {} seed(s) in {:.1f}s".format(dataset_name, variant, len(per_seed), time.time() - t0))

        for a, b, label in COMPARISONS:
            seeds_a = per_variant_per_seed_ndcg.get(a, [])
            seeds_b = per_variant_per_seed_ndcg.get(b, [])
            n_pairs = min(len(seeds_a), len(seeds_b))
            if n_pairs == 0:
                print("SKIP {} {}: no seed pairs for {} vs {}".format(dataset_name, label, a, b), file=sys.stderr)
                continue
            for group in GROUPS:
                pooled = {}
                for i in range(n_pairs):
                    ndcg_a, ndcg_b = seeds_a[i][group], seeds_b[i][group]
                    common_users = set(ndcg_a) & set(ndcg_b)
                    for user in common_users:
                        pooled.setdefault(user, []).append(ndcg_a[user] - ndcg_b[user])
                diffs = pool_diffs_across_seeds(pooled)
                stat = bootstrap(diffs, args.n_boot, rng)
                stat.update({
                    "dataset": dataset_name,
                    "comparison": "{}-{}".format(a, b),
                    "label": label,
                    "group": group,
                    "n_seed_pairs": n_pairs,
                    "metric": "NDCG@20",
                })
                results.append(stat)
                print(json.dumps(stat, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved {} rows to {}".format(len(results), args.output))


if __name__ == "__main__":
    main()
