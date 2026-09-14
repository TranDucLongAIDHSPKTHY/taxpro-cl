"""Per-user bootstrap CI for the warm-start-removal companion check (main
paper Section 5.4, RQ5 factorial): does removing warm_start_epochs entirely
(-> 0) change Recall@20, under both the random-direction (V0) and
taxonomy-direction (V3) controls, Amazon-Book, 3 seeds?

Previously only reported as a raw % point-estimate change (+18.90% random,
+21.38% taxonomy) with no per-user CI.
Each user's diff is averaged across the 3 seed pairs it appears in BEFORE
bootstrapping, matching tools/analysis/seed_matched_bootstrap.py's
methodology (and the A2 fix applied to the other factorial-adjacent
scripts).
"""
from __future__ import annotations
import json, logging, sys
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

PAIRS = {
    "random_direction": ("log/p0/taxprocl/amazon-book/A2-V0-nowarm", "log/p0/taxprocl/amazon-book/A2-V0"),
    "taxonomy_direction": ("log/p0/taxprocl/amazon-book/A2-V3-nowarm", "log/p0/taxprocl/amazon-book/A2-V3"),
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
    print("device:", device, flush=True)
    logging.basicConfig(level=logging.WARNING)
    protocol_dir = evaluation_protocol_dir(DATASET)
    targets_by_group = load_targets(protocol_dir, "test")

    results = {}
    for name, (nowarm_dir, warm_dir) in PAIRS.items():
        per_user = {g: {} for g in GROUPS}
        for seed in SEEDS:
            print(f"=== {name} seed {seed} ===", flush=True)
            m_nw, ds_nw, _c, _n = load_model(ROOT / nowarm_dir / f"seed{seed}", device)
            pu_nw = per_user_recall(m_nw, ds_nw, device, targets_by_group)
            del m_nw
            m_w, ds_w, _c, _n = load_model(ROOT / warm_dir / f"seed{seed}", device)
            pu_w = per_user_recall(m_w, ds_w, device, targets_by_group)
            del m_w
            for g in GROUPS:
                for u in set(pu_nw[g]) & set(pu_w[g]):
                    per_user[g].setdefault(u, []).append(pu_nw[g][u] - pu_w[g][u])
        pooled = {g: [float(np.mean(vals)) for vals in per_user[g].values()] for g in GROUPS}
        rng = np.random.default_rng(42)
        name_results = {}
        print(f"-- {name}: nowarm minus warm --")
        for g in GROUPS:
            stat = bootstrap(pooled[g], N_BOOT, rng)
            name_results[g] = stat
            print(f"{g}: n={stat['n_users']} mean_diff={stat['mean_diff']:+.6f} "
                  f"95% CI=[{stat['ci95_lo']:+.6f}, {stat['ci95_hi']:+.6f}] excludes_zero={stat['excludes_zero']}")
        results[name] = name_results

    out = ROOT / "results" / "warmstart_removal_bootstrap.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved to", out)


if __name__ == "__main__":
    main()
