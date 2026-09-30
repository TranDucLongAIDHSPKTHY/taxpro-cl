"""Strict-Cold audit of the Protocol A main-comparison checkpoints
(Online Resource 1, Section S34; main paper Limitation 3 and Table 5).

Strict-Cold items (train degree 0) are isolated in the training graph. Under
layer-0-excluding pooling (SimGCL, XSimGCL, TaxPro-CL) their propagated
embedding is all-zero, so every user's dot-product score is exactly 0; under
layer-0-including pooling (LightGCN, SGL-ED, NCL) they keep a scaled layer-0
lookup embedding. For every method x {amazon-book, yelp2018} x seed listed as
"<model>-main" in results/results_manifest.csv, this re-scores the test split
(full catalog, train and validation items excluded, K=20) and reports:
Strict-Cold items in any top-20, Strict-Cold Recall@20, ties at the K-th
position (and ties at a zero score), the largest |score| of a Strict-Cold item,
and Overall Recall@20 with and without Strict-Cold positives.

Inference only, no retraining.

Usage:
    python -m tools.analysis.strict_cold_audit [--device cuda] [--models LightGCN ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir  # noqa: E402
from utility.utility_train.group_evaluator import load_targets  # noqa: E402
from tools.ranking.inference import load_model  # noqa: E402

DATASETS = ("amazon-book", "yelp2018")
K = 20
BATCH_SIZE = 256


def main_runs():
    runs = {}
    with open(ROOT / "results" / "results_manifest.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] in DATASETS and row["variant"].endswith("-main"):
                runs.setdefault((row["model"], row["dataset"]), []).append(
                    (row["seed"], row["run_dir"].replace("\\", "/")))
    return runs


def audit(model, dataset, strict_cold_mask, strict_cold_items, targets, device):
    users = sorted(targets)
    n_top = users_with = ties = ties_zero = 0
    score_absmax = 0.0
    recall_all, recall_without, recall_sc = [], [], []
    for start in range(0, len(users), BATCH_SIZE):
        batch = users[start:start + BATCH_SIZE]
        with torch.no_grad():
            rating = model.get_rating_for_test(torch.tensor(batch, device=device))
            score_absmax = max(score_absmax, float(rating[:, strict_cold_mask].abs().max()))
            excluded = dataset.get_user_pos_items_for_evaluation(batch, "test")
            for row, items in enumerate(excluded):
                if len(items):
                    rating[row, list(items)] = float("-inf")
            values, index = torch.topk(rating, K + 1, dim=1)
        top = index[:, :K]
        in_sc = strict_cold_mask[top]
        n_top += int(in_sc.sum())
        users_with += int(in_sc.any(1).sum())
        tie = values[:, K - 1] == values[:, K]
        ties += int(tie.sum())
        # the framework's rating is sigmoid(score): a zero dot product is 0.5
        ties_zero += int((tie & (values[:, K - 1] == 0.5)).sum())
        top_np = top.cpu().numpy()
        for row, user in enumerate(batch):
            positives = set(targets[user])
            top_set = set(top_np[row].tolist())
            recall_all.append(len(positives & top_set) / len(positives))
            rest = positives - strict_cold_items
            if rest:
                recall_without.append(len(rest & top_set) / len(rest))
            sc = positives & strict_cold_items
            if sc:
                recall_sc.append(len(sc & top_set) / len(sc))
    return {
        "n_users": len(users), "sc_items_in_top20_total": n_top, "users_with_sc_in_top20": users_with,
        "ties_at_K": ties, "ties_at_K_zero_score": ties_zero, "sc_rating_absmax": score_absmax,
        "overall_recall20": float(np.mean(recall_all)),
        "overall_recall20_excl_sc_pos": float(np.mean(recall_without)), "n_users_excl_sc": len(recall_without),
        "strict_cold_recall20": float(np.mean(recall_sc)) if recall_sc else None, "n_users_sc": len(recall_sc),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "strict_cold_audit.json")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    out = {}
    t0 = time.time()
    for (model_name, dataset_name), runs in sorted(main_runs().items()):
        if args.models and model_name not in args.models:
            continue
        protocol = Path(evaluation_protocol_dir(dataset_name))
        mask_np = np.load(protocol / "group_masks.npz")["strict_cold"]
        mask = torch.as_tensor(mask_np, device=device)
        items = set(np.nonzero(mask_np)[0].tolist())
        targets = load_targets(protocol, "test")["overall"]
        for seed, run_dir in sorted(runs):
            model, dataset, _config, _name = load_model(ROOT / run_dir, device)
            key = f"{model_name}|{dataset_name}|{seed}"
            out[key] = {"run_dir": run_dir, **audit(model, dataset, mask, items, targets, device)}
            print(key, out[key], f"t={time.time() - t0:.0f}s", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("Saved to", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
