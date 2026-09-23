"""GVHD V6 review, points A1 and A2: (1) restore the Amazon-Book no_merge
V0-V3 factorial (dropped from the manuscript when the team reran under
merge_t10, even though both are valid controlled comparisons under different
taxonomy policies), and (2) multiplicity-correct the direction and
epsilon-adaptivity factorial cells with the same paired sign-flip test and
Holm-Bonferroni procedure the manuscript already applies to Table 11
(tools/analysis/a2_multiplicity_correction.py), instead of leaving the
factorial's bootstrap CIs uncorrected.

This script does not modify or re-run anything that already produces a
published table (factorial_direction_bootstrap.py / S13,S17,S19,Fig.3;
a2_mergedt10_factorial_recompute.py / results/a2_mergedt10_factorial_bootstrap.json).
It rescoring the same checkpoints those scripts already use (inference only,
no retraining) plus the original Amazon-Book no_merge V0-V3 checkpoints
(log/p0/taxprocl/amazon-book/A2-V{0,1,2,3}, still on disk, 3 seeds,
config_resolved.json confirms taxonomy_policy=no_merge, environment.json
confirms all four ran on the same machine, Environment A / RTX 4060 Ti).

For each dataset x checkpoint-set (the "current" one behind S13/S17/S19, and,
Amazon-Book only, the "no_merge" sensitivity set), it computes, per user,
Recall@20 and NDCG@20 for near_cold/long_tail/overall/warm, then:
  - the four marginal comparisons (V1-V0, V3-V2: direction; V2-V0, V3-V1:
    epsilon-adaptivity), per-user diffs pooled across the 3 seeds (mean
    across the seeds a user appears in -- identical convention to
    factorial_direction_bootstrap.py's pool_diffs_across_seeds);
  - a percentile bootstrap 95% CI (5000 resamples), matching S13/S17/S19;
  - a paired sign-flip randomization p-value (10^5 permutations, seed 42;
    reusing the exact function from a2_multiplicity_correction.py), plus a
    10^6-permutation / seed-123 precision re-run, matching S26's practice;
  - for each dataset x group, an intersection-union combined p
    (p_cell = max(p_V1-V0, p_V3-V2) for direction; max(p_V2-V0, p_V3-V1) for
    epsilon-adaptivity) -- the two comparisons of one cell share users and
    seeds, so this is not two independent tests, matching the manuscript's
    own decision rule (both comparisons must favor taxonomy);
  - Holm-Bonferroni (alpha=0.05) across the 8 dataset x group (near_cold,
    long_tail only -- the same family structure as Table 11/S26) combined
    p-values, separately for direction and epsilon-adaptivity, on Recall@20
    (the metric Table 11/the Abstract's headline claim uses) and, as a
    secondary check, on NDCG@20.

Usage:
    python -m tools.analysis.a1a2_factorial_multiplicity
    python -m tools.analysis.a1a2_factorial_multiplicity --skip-precision-rerun
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

SEEDS = ["42", "0", "1"]
GROUPS = ["near_cold", "long_tail", "overall", "warm"]
HOLM_GROUPS = ["near_cold", "long_tail"]  # Table 11 / S26's family structure
K = 20
BATCH_SIZE = 256
N_BOOT = 5000
N_PERM_PRIMARY = 100_000
N_PERM_PRECISION = 1_000_000
PRECISION_SEED = 123

# Checkpoint sets. "current" = exactly what S13/S17/S19/Fig.3 already report
# (factorial_direction_bootstrap.py's DATASET_DIRS, reproduced verbatim so
# this script's "current" numbers must match the published ones).
# "nomerge" is Amazon-Book only: the original no_merge V0-V3 run, dropped
# from the manuscript body when the team reran under merge_t10 (A1).
CHECKPOINT_SETS = {
    "amazon-book": {
        "current": {
            "V0": "log/p0/taxprocl/amazon-book/A2-V0-mergedt10",
            "V1": "log/p0/taxprocl/amazon-book/A2-V1-mergedt10",
            "V2": "log/p0/taxprocl/amazon-book/A2-V2-mergedt10",
            "V3": "log/p0/taxprocl/amazon-book/A2-V3-warm20",
        },
        "nomerge": {
            "V0": "log/p0/taxprocl/amazon-book/A2-V0",
            "V1": "log/p0/taxprocl/amazon-book/A2-V1",
            "V2": "log/p0/taxprocl/amazon-book/A2-V2",
            "V3": "log/p0/taxprocl/amazon-book/A2-V3",
        },
    },
    "yelp2018": {
        "current": {
            "V0": "log/p0/taxprocl/yelp2018/A2-V0-tempuser0.15",
            "V1": "log/p0/taxprocl/yelp2018/A2-V1-tempuser0.15",
            "V2": "log/p0/taxprocl/yelp2018/A2-V2-tempuser0.15",
            "V3": "log/p0/taxprocl/yelp2018/A2-V3-tempuser0.15",
        },
    },
    "musical-instruments": {
        "current": {
            "V0": "log/p0/taxprocl/musical-instruments/A2-V0",
            "V1": "log/p0/taxprocl/musical-instruments/A2-V1",
            "V2": "log/p0/taxprocl/musical-instruments/A2-V2",
            "V3": "log/p0/taxprocl/musical-instruments/taxpro-cl-FINAL-no_merge-temp0.15-gammacold5.0",
        },
    },
    "arts-crafts-and-sewing": {
        "current": {
            "V0": "log/p0/taxprocl/arts-crafts-and-sewing/A2-V0",
            "V1": "log/p0/taxprocl/arts-crafts-and-sewing/A2-V1",
            "V2": "log/p0/taxprocl/arts-crafts-and-sewing/A2-V2",
            "V3": "log/p0/taxprocl/arts-crafts-and-sewing/taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0",
        },
    },
}

COMPARISONS = {
    "direction_fixed_eps__V1-V0": [(+1, "V1"), (-1, "V0")],
    "direction_adaptive_eps__V3-V2": [(+1, "V3"), (-1, "V2")],
    "eps_random_dir__V2-V0": [(+1, "V2"), (-1, "V0")],
    "eps_taxonomy_dir__V3-V1": [(+1, "V3"), (-1, "V1")],
}
FACTOR_OF = {
    "direction_fixed_eps__V1-V0": "direction",
    "direction_adaptive_eps__V3-V2": "direction",
    "eps_random_dir__V2-V0": "epsilon",
    "eps_taxonomy_dir__V3-V1": "epsilon",
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


STATS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bootstrap_ci(diffs, n_boot, rng, chunk=1000):
    """GPU-vectorized percentile bootstrap (mathematically identical
    resampling scheme to the CPU per-iteration loop in
    a2_mergedt10_factorial_recompute.py / factorial_direction_bootstrap.py,
    just batched on the idle GPU instead of a 5000-iteration Python loop --
    this analysis is new, not reproducing an already-published number
    bit-for-bit, so torch's own RNG stream is fine)."""
    arr = torch.as_tensor(np.asarray(diffs, dtype=np.float64), device=STATS_DEVICE)
    n = arr.shape[0]
    seed = int(rng.integers(0, 2**31 - 1))
    g = torch.Generator(device=STATS_DEVICE)
    g.manual_seed(seed)
    boot_parts = []
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = torch.randint(0, n, (m, n), generator=g, device=STATS_DEVICE)
        boot_parts.append(arr[idx].mean(dim=1))
        done += m
    boot = torch.cat(boot_parts).sort().values
    lo = float(boot[int(0.025 * n_boot)].item())
    hi = float(boot[int(0.975 * n_boot) - 1].item())
    return {"n_users": n, "mean_diff": float(arr.mean().item()), "ci95_lo": lo, "ci95_hi": hi,
            "excludes_zero": bool((lo > 0) or (hi < 0))}


def sign_flip_pvalue(values, n_perm, rng, chunk=2000):
    """Same paired sign-flip randomization test as
    tools/analysis/a2_multiplicity_correction.py (mathematically identical:
    two-sided, floored at 1/(n_perm+1)), batched on GPU."""
    arr = torch.as_tensor(np.asarray(values, dtype=np.float64), device=STATS_DEVICE)
    n = arr.shape[0]
    observed = arr.sum().abs().item()
    seed = int(rng.integers(0, 2**31 - 1))
    g = torch.Generator(device=STATS_DEVICE)
    g.manual_seed(seed)
    extreme = 0
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = (torch.randint(0, 2, (m, n), generator=g, device=STATS_DEVICE, dtype=torch.int8).double() * 2 - 1)
        sums = (signs @ arr).abs()
        extreme += int((sums >= observed - 1e-12).sum().item())
        done += m
    return (extreme + 1) / (n_perm + 1)


def holm_bonferroni(cells, p_key, out_key, alpha=0.05):
    """Verbatim from tools/analysis/a2_multiplicity_correction.py."""
    order = sorted(range(len(cells)), key=lambda i: cells[i][p_key])
    m = len(cells)
    still_significant = True
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        significant = still_significant and cells[idx][p_key] <= threshold
        still_significant = significant
        cells[idx][out_key] = bool(significant)
        cells[idx][out_key + "_threshold"] = threshold
    return cells


def score_checkpoint_set(dataset, rel_dirs, device):
    protocol_dir = evaluation_protocol_dir(dataset)
    targets_by_group = load_targets(protocol_dir, "test")
    scores = {}
    for variant, rel in rel_dirs.items():
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
    return scores


def pooled_diffs(scores, terms, metric, group):
    pooled = {}
    for seed in SEEDS:
        users = set.intersection(*[set(scores[v][seed][metric][group]) for _, v in terms])
        for u in users:
            d = sum(sign * scores[v][seed][metric][group][u] for sign, v in terms)
            pooled.setdefault(u, []).append(d)
    return {u: float(np.mean(v)) for u, v in pooled.items()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", nargs="+", default=list(CHECKPOINT_SETS))
    parser.add_argument("--skip-precision-rerun", action="store_true",
                         help="skip the 10^6-permutation / seed-123 precision check")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "a1a2_factorial_multiplicity.json")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    t0 = time.time()

    out = {"comparisons_meaning": {k: FACTOR_OF[k] for k in COMPARISONS}, "cells": {}}

    for dataset in args.datasets:
        for ckpt_set_name, rel_dirs in CHECKPOINT_SETS[dataset].items():
            print(f"\n########## {dataset} / {ckpt_set_name} ########## elapsed={time.time()-t0:.0f}s", flush=True)
            scores = score_checkpoint_set(dataset, rel_dirs, device)
            key_prefix = f"{dataset}|{ckpt_set_name}"

            for cmp_name, terms in COMPARISONS.items():
                for metric in ("recall", "ndcg"):
                    for g in GROUPS:
                        diffs_by_user = pooled_diffs(scores, terms, metric, g)
                        diffs = list(diffs_by_user.values())
                        rng = np.random.default_rng(42)
                        ci = bootstrap_ci(diffs, N_BOOT, rng)
                        p_flip = sign_flip_pvalue(diffs, N_PERM_PRIMARY, np.random.default_rng(42))
                        cell = {**ci, "p_signflip_1e5_seed42": p_flip,
                                "dataset": dataset, "checkpoint_set": ckpt_set_name,
                                "comparison": cmp_name, "factor": FACTOR_OF[cmp_name],
                                "metric": metric, "group": g}
                        if not args.skip_precision_rerun:
                            cell["p_signflip_1e6_seed123"] = sign_flip_pvalue(
                                diffs, N_PERM_PRECISION, np.random.default_rng(PRECISION_SEED))
                        out["cells"].setdefault(key_prefix, {})[f"{cmp_name}|{metric}|{g}"] = cell
                        print(f"  {cmp_name:28s} {metric:6s} {g:10s} n={ci['n_users']:6d} "
                              f"diff={ci['mean_diff']:+.6f} CI=[{ci['ci95_lo']:+.6f},{ci['ci95_hi']:+.6f}] "
                              f"p_flip={p_flip:.5f}", flush=True)

    # Combine each dataset x checkpoint_set x group's two same-factor
    # comparisons via intersection-union (p = max), then Holm across the
    # 8 dataset x {near_cold,long_tail} cells, separately per factor,
    # separately per checkpoint_set label ("current" pools across all 4
    # datasets' "current" set; "nomerge" is Amazon-Book only, reported but
    # not Holm-corrected on its own since it is a single-dataset sensitivity
    # check, not a new family).
    out["combined_cells"] = {}
    for metric in ("recall", "ndcg"):
        for factor, pair in (("direction", ("direction_fixed_eps__V1-V0", "direction_adaptive_eps__V3-V2")),
                              ("epsilon", ("eps_random_dir__V2-V0", "eps_taxonomy_dir__V3-V1"))):
            family = []
            for dataset in args.datasets:
                for ckpt_set_name in CHECKPOINT_SETS[dataset]:
                    key_prefix = f"{dataset}|{ckpt_set_name}"
                    for g in HOLM_GROUPS:
                        c1 = out["cells"][key_prefix][f"{pair[0]}|{metric}|{g}"]
                        c2 = out["cells"][key_prefix][f"{pair[1]}|{metric}|{g}"]
                        combined = {
                            "dataset": dataset, "checkpoint_set": ckpt_set_name, "factor": factor,
                            "metric": metric, "group": g,
                            "p_signflip_1e5_seed42": max(c1["p_signflip_1e5_seed42"], c2["p_signflip_1e5_seed42"]),
                            "mean_diff_v1v0_or_v2v0": c1["mean_diff"], "mean_diff_v3v2_or_v3v1": c2["mean_diff"],
                        }
                        if "p_signflip_1e6_seed123" in c1:
                            combined["p_signflip_1e6_seed123"] = max(
                                c1["p_signflip_1e6_seed123"], c2["p_signflip_1e6_seed123"])
                        out["combined_cells"].setdefault(f"{key_prefix}|{factor}|{metric}", {})[g] = combined
                        if ckpt_set_name == "current":
                            family.append(combined)
            if len(family) == len(args.datasets) * len(HOLM_GROUPS):
                family = holm_bonferroni(family, "p_signflip_1e5_seed42", "holm_significant", alpha=0.05)
                if not args.skip_precision_rerun:
                    family = holm_bonferroni(family, "p_signflip_1e6_seed123", "holm_significant_precision", alpha=0.05)
                print(f"\n=== Holm-Bonferroni, factor={factor} metric={metric}, "
                      f"family={[ (c['dataset'], c['group']) for c in family ]} ===")
                for c in sorted(family, key=lambda x: x["p_signflip_1e5_seed42"]):
                    print(f"  {c['dataset']:25s} {c['group']:10s} p={c['p_signflip_1e5_seed42']:.5f} "
                          f"thr={c['holm_significant_threshold']:.5f} holm_sig={c['holm_significant']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nSaved to", args.output, f"elapsed={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
