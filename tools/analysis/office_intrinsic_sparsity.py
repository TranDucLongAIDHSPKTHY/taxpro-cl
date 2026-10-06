"""Held-out (Office-Products) re-analysis of the primary comparison restricted to items sparse under both degree
definitions (train-time degree and pool degree over train+validation+test, Protocol B), the definition of Online
Resource 1, Table S25. Same checkpoints as main paper Table 12 (selected TaxPro-CL against the rule-R and default
SimGCL, three seed-matched pairs); per-user Recall@20 differences are averaged over the seed pairs, then
resampled over users (bootstrap, 2,000 resamples, seed 42) and tested with the paired sign-flip test (10^5 sign
assignments, seed 42). Analysis after the test split was opened; not part of the prospectively specified protocol.

Inference only, no retraining. Usage: python -m tools.analysis.office_intrinsic_sparsity [--device cpu]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir, verified_dataset_dir  # noqa: E402
from tools.ranking import inference  # noqa: E402
from utility.utility_train.group_evaluator import load_targets  # noqa: E402

DATASET = "office-products"
K = 20
COMPARATORS = {"tuned_R": "HO-SimGCL-tuned-R", "default": "HO-SimGCL-default"}


def degrees():
    out = {}
    for split in ("train", "validation", "test"):
        c = Counter()
        with open(Path(verified_dataset_dir(DATASET)) / f"{split}.txt", encoding="utf-8") as f:
            for line in f:
                c.update(int(i) for i in line.split()[1:])
        out[split] = c
    pool = out["train"] + out["validation"] + out["test"]
    return out["train"], pool


def runs(variant):
    d = {}
    with open(ROOT / "results" / "results_manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == DATASET and r["variant"] == variant:
                d[int(r["seed"])] = ROOT / r["run_dir"].replace("\\", "/")
    return d


def per_user_recall(run_dir, targets_by_group, device, batch=512):
    model, dataset, _, _ = inference.load_model(run_dir, device)
    users = sorted(set().union(*[set(t) for t in targets_by_group.values()]))
    out = {g: {} for g in targets_by_group}
    for s in range(0, len(users), batch):
        b = users[s:s + batch]
        with torch.no_grad():
            rating = model.get_rating_for_test(torch.tensor(b, device=device))
            for row, items in enumerate(dataset.get_user_pos_items_for_evaluation(b, "test")):
                if len(items):
                    rating[row, list(items)] = float("-inf")
            top = torch.topk(rating, K, dim=1).indices.cpu().numpy()
        for row, u in enumerate(b):
            ts = set(top[row].tolist())
            for g, t in targets_by_group.items():
                if u in t:
                    out[g][u] = len(ts & set(t[u])) / len(t[u])
    return out


def bootstrap(values, n_boot=2000, seed=42):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.sort(np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]))
    return float(means[int(0.025 * n_boot)]), float(means[int(0.975 * n_boot) - 1])


def sign_flip(values, n_perm=100_000, seed=42, chunk=1000):
    v = np.asarray(values, dtype=np.float64)
    obs = abs(v.sum())
    rng = np.random.default_rng(seed)
    ext = done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.integers(0, 2, size=(m, len(v)), dtype=np.int8) * 2 - 1
        ext += int((np.abs(signs @ v) >= obs - 1e-12).sum())
        done += m
    return (ext + 1) / (n_perm + 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "confirmatory" / "intrinsic_sparsity_office-products.json")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    train, pool = degrees()
    nc_train = [i for i, d in train.items() if 1 <= d <= 5]
    nc_items = {i for i in nc_train if 1 <= pool[i] <= 5}
    lt_items = {i for i, d in train.items() if 1 <= d <= 10 and 1 <= pool[i] <= 10}
    tg = load_targets(evaluation_protocol_dir(DATASET), "test")
    targets = {}
    for g, allowed in (("near_cold", nc_items), ("long_tail", lt_items)):
        targets[g] = {u: [i for i in its if i in allowed] for u, its in tg[g].items()}
        targets[g] = {u: v for u, v in targets[g].items() if v}
    res = {"dataset": DATASET,
           "near_cold_train_items": len(nc_train),
           "near_cold_split_induced_share": sum(1 for i in nc_train if pool[i] > 5) / len(nc_train),
           "pool_confirmed_items": {"near_cold": len(nc_items), "long_tail": len(lt_items)},
           "eligible_users": {g: len(t) for g, t in targets.items()}, "comparisons": {}}
    tax = runs("HO-TaxPro-CL-selected")
    scores = {s: per_user_recall(tax[s], targets, device) for s in sorted(tax)}
    for name, variant in COMPARATORS.items():
        comp = runs(variant)
        cs = {s: per_user_recall(comp[s], targets, device) for s in sorted(comp)}
        res["comparisons"][name] = {}
        for g in ("near_cold", "long_tail"):
            users = sorted(targets[g])
            diffs = [float(np.mean([scores[s][g][u] - cs[s][g][u] for s in sorted(tax)])) for u in users]
            comp_mean = float(np.mean([np.mean([cs[s][g][u] for u in users]) for s in sorted(comp)]))
            lo, hi = bootstrap(diffs)
            res["comparisons"][name][g] = {
                "n_users": len(users), "mean_diff": float(np.mean(diffs)), "ci95": [lo, hi],
                "comparator_mean": comp_mean, "relative": float(np.mean(diffs)) / comp_mean if comp_mean else None,
                "p_signflip": sign_flip(diffs),
                "per_seed_diff": {s: float(np.mean([scores[s][g][u] - cs[s][g][u] for u in users])) for s in sorted(tax)},
            }
            print(name, g, res["comparisons"][name][g], flush=True)
    args.output.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="")
    print("Saved to", args.output)


if __name__ == "__main__":
    main()
