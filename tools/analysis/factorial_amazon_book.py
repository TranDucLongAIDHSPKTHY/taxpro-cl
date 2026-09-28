"""Amazon-Book direction-by-epsilon-adaptivity factorial (RQ5), recomputed
on the merge_t10 checkpoints (the main configuration's actual taxonomy
policy), replacing the earlier no_merge V0-V3 checkpoints.

Checkpoints (all taxonomy_policy=merge_t10, temperature 0.1, warm_start 20,
identical config except augmentation_direction / use_adaptive_epsilon):
    V0 = A2-V0-mergedt10  (random,   fixed epsilon)
    V1 = A2-V1-mergedt10  (taxonomy, fixed epsilon)
    V2 = A2-V2-mergedt10  (random,   adaptive epsilon)
    V3 = A2-V3-warm20     (taxonomy, adaptive epsilon; an independent run of
                           the main configuration on other hardware --
                           RTX 3090 Ti, commit a20d2e6 -- whose test metrics
                           are within 0.26% of the TaxPro-CL-main
                           checkpoint's; pass --v3-dir to use the main
                           checkpoint as V3 instead, as a sensitivity check)
Warm-start-removal companion check (same policy on both sides):
    random    : A2-V0-nowarm  vs A2-V0-mergedt10
    taxonomy  : A2-V3-nowarm  vs A2-V3-warm20

Each checkpoint is scored once per seed; Recall@20 and NDCG@20 per user for
Near-Cold / Long-Tail / Overall / Warm. Every difference is averaged per
user across the three seeds BEFORE bootstrap-resampling users (same
procedure as factorial_direction_bootstrap.py; 5000 resamples, percentile
method). The interaction term (V3-V2)-(V1-V0) is computed per user from the
same data, not from two independent bootstraps.

Inference only; no retraining.

Usage:
    python -m tools.analysis.a2_mergedt10_factorial_recompute
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

from config_path.config_path import evaluation_protocol_dir
from utility.utility_train.group_evaluator import load_targets
from tests.Recommendation_system.inference import load_model, compute_batch_order_and_rank

DATASET = "amazon-book"
BASE = "log/p0/taxprocl/amazon-book/"
SEEDS = ["42", "0", "1"]
GROUPS = ["near_cold", "long_tail", "overall", "warm"]
K = 20
BATCH_SIZE = 256
N_BOOT = 5000

CHECKPOINTS = {
    "V0": BASE + "A2-V0-mergedt10",
    "V1": BASE + "A2-V1-mergedt10",
    "V2": BASE + "A2-V2-mergedt10",
    "V3": BASE + "A2-V3-warm20",
    "V0_nowarm": BASE + "A2-V0-nowarm",
    "V3_nowarm": BASE + "A2-V3-nowarm",
}

# name -> (minuend expression, subtrahend expression), each a list of
# (sign, variant) terms; a difference is sum(sign * per-user metric).
COMPARISONS = {
    "direction_fixed_eps__V1-V0": [(+1, "V1"), (-1, "V0")],
    "direction_adaptive_eps__V3-V2": [(+1, "V3"), (-1, "V2")],
    "eps_random_dir__V2-V0": [(+1, "V2"), (-1, "V0")],
    "eps_taxonomy_dir__V3-V1": [(+1, "V3"), (-1, "V1")],
    "interaction__(V3-V2)-(V1-V0)": [(+1, "V3"), (-1, "V2"), (-1, "V1"), (+1, "V0")],
    "nowarm_random__V0nowarm-V0": [(+1, "V0_nowarm"), (-1, "V0")],
    "nowarm_taxonomy__V3nowarm-V3": [(+1, "V3_nowarm"), (-1, "V3")],
}


def per_user_metrics(model, dataset, device, targets_by_group):
    all_users = sorted(set().union(*[set(t.keys()) for t in targets_by_group.values()]))
    rec = {g: {} for g in GROUPS}
    ndcg = {g: {} for g in GROUPS}
    discounts = 1.0 / np.log2(np.arange(K) + 2)
    for start in range(0, len(all_users), BATCH_SIZE):
        batch = all_users[start:start + BATCH_SIZE]
        order, _rank = compute_batch_order_and_rank(model, dataset, device, batch, split="test")
        order_np = order[:, :K].cpu().numpy()
        for row_idx, user in enumerate(batch):
            topk = order_np[row_idx].tolist()
            for g in GROUPS:
                positives = targets_by_group[g].get(user)
                if not positives:
                    continue
                pos = set(positives)
                hit = np.array([1.0 if it in pos else 0.0 for it in topk])
                rec[g][user] = float(hit.sum() / len(pos))
                idcg = float(discounts[:min(K, len(pos))].sum())
                if idcg > 0:
                    ndcg[g][user] = float((hit * discounts).sum() / idcg)
    return {"recall": rec, "ndcg": ndcg}


def bootstrap(diffs, n_boot, rng):
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = arr[rng.integers(0, n, size=n)].mean()
    boot.sort()
    lo = float(boot[int(0.025 * n_boot)])
    hi = float(boot[int(0.975 * n_boot) - 1])
    return {"n_users": n, "mean_diff": float(arr.mean()), "ci95_lo": lo, "ci95_hi": hi,
            "excludes_zero": bool((lo > 0) or (hi < 0))}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "a2_mergedt10_factorial_bootstrap.json")
    parser.add_argument("--v3-dir", default=None,
                        help="alternative V3 run directory (relative to project root); with this option the "
                             "warm-start-removal comparisons are skipped")
    args = parser.parse_args(argv)
    checkpoints = dict(CHECKPOINTS)
    comparisons = dict(COMPARISONS)
    if args.v3_dir:
        checkpoints["V3"] = args.v3_dir
        checkpoints.pop("V0_nowarm"); checkpoints.pop("V3_nowarm")
        comparisons = {k: v for k, v in comparisons.items() if not k.startswith("nowarm")}

    t0 = time.time()
    device = torch.device(args.device)
    targets = load_targets(evaluation_protocol_dir(DATASET), "test")
    targets_by_group = {g: targets[g] for g in GROUPS}

    # scores[variant][seed] = {"recall": {group: {user: v}}, "ndcg": {...}}
    scores = {}
    for variant, rel in checkpoints.items():
        scores[variant] = {}
        for seed in SEEDS:
            run_dir = ROOT / rel / f"seed{seed}"
            if not (run_dir / "run_manifest.json").is_file():
                raise SystemExit(f"missing checkpoint: {run_dir}")
            model, ds, _cfg, _name = load_model(run_dir, device)
            scores[variant][seed] = per_user_metrics(model, ds, device, targets_by_group)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[scored] {variant} elapsed={time.time()-t0:.0f}s", flush=True)

    out = {"dataset": DATASET, "checkpoints": checkpoints, "n_boot": args.n_boot,
           "seeds": SEEDS, "point_estimates": {}, "comparisons": {}}

    # point estimates: seed-mean of the per-user-mean metric, per variant/group
    for variant in checkpoints:
        out["point_estimates"][variant] = {}
        for metric in ("recall", "ndcg"):
            out["point_estimates"][variant][metric] = {}
            for g in GROUPS:
                per_seed = [float(np.mean(list(scores[variant][s][metric][g].values()))) for s in SEEDS]
                out["point_estimates"][variant][metric][g] = {
                    "per_seed": per_seed, "mean": float(np.mean(per_seed)),
                    "std": float(np.std(per_seed, ddof=1))}

    rng = np.random.default_rng(42)
    for name, terms in comparisons.items():
        out["comparisons"][name] = {}
        for metric in ("recall", "ndcg"):
            out["comparisons"][name][metric] = {}
            for g in GROUPS:
                pooled = {}
                for seed in SEEDS:
                    users = set.intersection(*[set(scores[v][seed][metric][g]) for _, v in terms])
                    for u in users:
                        d = sum(sign * scores[v][seed][metric][g][u] for sign, v in terms)
                        pooled.setdefault(u, []).append(d)
                diffs = [float(np.mean(v)) for v in pooled.values()]
                stat = bootstrap(diffs, args.n_boot, rng)
                out["comparisons"][name][metric][g] = stat
                print(f"{name:34s} {metric:6s} {g:10s} n={stat['n_users']:6d} "
                      f"diff={stat['mean_diff']:+.6f} CI=[{stat['ci95_lo']:+.6f},{stat['ci95_hi']:+.6f}] "
                      f"excl0={stat['excludes_zero']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Saved to", args.output, f"elapsed={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
