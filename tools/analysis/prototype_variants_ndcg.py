"""B1: NDCG@20 for the prototype-
construction sensitivity variants (leaf_uniform, leave_one_out, rescue vs.
TaxPro-CL-main), Amazon-Book, extending Section S14 (Table S14,
Recall@20-only) to NDCG@20, mirroring b1_factorial_ndcg.py's treatment of
the direction factorial. Same checkpoints (3 seeds: 0, 1, 42) and per-user
pooling-across-seeds procedure as tools/analysis/b2_prototype_variant_overlap.py
and the original Table S14 Recall@20 computation.

Inference only; no retraining; no new checkpoints created.
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

from config_path.config_path import RESULT_DIR, evaluation_protocol_dir
from utility.utility_train.group_evaluator import load_targets
from tests.Recommendation_system.inference import compute_batch_order_and_rank
from tools.analysis.b2_prototype_variant_overlap import (
    BASE, MAIN_DIR_NAME, VARIANTS, SEEDS, K,
    load_model_dropping_stale_buffers,
)

DATASET = "amazon-book"
GROUPS = ["near_cold", "long_tail", "overall", "warm"]
BATCH_SIZE = 256
N_BOOT = 5000


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(42)
    t0 = time.time()

    protocol_dir = evaluation_protocol_dir(DATASET)
    targets_by_group = {g: load_targets(protocol_dir, "test")[g] for g in GROUPS}

    per_variant_per_seed_ndcg = {}
    for variant_name, variant_subdir in list(VARIANTS.items()) + [("main", None)]:
        per_seed = []
        for seed in SEEDS:
            run_dir = (BASE / variant_subdir / "seed{}".format(seed)) if variant_subdir else (BASE / MAIN_DIR_NAME / "seed{}".format(seed))
            if not (run_dir / "run_manifest.json").is_file():
                print("SKIP {} seed={}: missing checkpoint at {}".format(variant_name, seed, run_dir), file=sys.stderr)
                continue
            model, ds, _cfg, _name = load_model_dropping_stale_buffers(run_dir, device)
            per_user = per_user_ndcg(model, ds, device, targets_by_group)
            per_seed.append(per_user)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        per_variant_per_seed_ndcg[variant_name] = per_seed
        print("{}: scored {} seed(s) in {:.1f}s".format(variant_name, len(per_seed), time.time() - t0))

    results = []
    for variant_name in VARIANTS:
        seeds_v = per_variant_per_seed_ndcg.get(variant_name, [])
        seeds_m = per_variant_per_seed_ndcg.get("main", [])
        n_pairs = min(len(seeds_v), len(seeds_m))
        if n_pairs == 0:
            print("SKIP {}: no seed pairs".format(variant_name), file=sys.stderr)
            continue
        for group in GROUPS:
            pooled = {}
            for i in range(n_pairs):
                ndcg_v, ndcg_m = seeds_v[i][group], seeds_m[i][group]
                common_users = set(ndcg_v) & set(ndcg_m)
                for user in common_users:
                    pooled.setdefault(user, []).append(ndcg_v[user] - ndcg_m[user])
            diffs = pool_diffs_across_seeds(pooled)
            stat = bootstrap(diffs, args.n_boot, rng)
            stat.update({
                "variant": variant_name,
                "comparison": "{}-main".format(variant_name),
                "group": group,
                "n_seed_pairs": n_pairs,
                "metric": "NDCG@20",
            })
            results.append(stat)
            print(json.dumps(stat, indent=2))

    output = RESULT_DIR / "b1_prototype_ndcg.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved {} rows to {}".format(len(results), output))


if __name__ == "__main__":
    main()
