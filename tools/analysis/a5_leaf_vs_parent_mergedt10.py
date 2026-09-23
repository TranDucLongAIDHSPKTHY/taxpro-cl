"""A5 (leaf vs. parent prototype granularity), Amazon-Book, policy-matched.

Compares TaxPro-CL-main (prototype_mode=leaf, taxonomy_policy=merge_t10) with
ablation-A5-parent-mergedt10 (prototype_mode=parent, taxonomy_policy=merge_t10;
launched by tools/experiments/run_a5_parent_mergedt10.ps1). The two differ in
prototype_mode only, unlike the earlier ablation-A5-parent run, which used
taxonomy_policy=no_merge.

For each of the three seeds (42, 0, 1) both checkpoints are scored once:
Recall@20 and NDCG@20 per user for Near-Cold / Long-Tail / Overall / Warm, and
the top-20 overlap between the two checkpoints over the Overall-eligible
users. The parent-minus-leaf difference is averaged per user across the three
seeds before bootstrap-resampling users (5000 resamples, percentile method,
seed 42), the procedure used for the RQ5 factorial.

Inference only; no retraining.

Usage:
    python -m tools.analysis.a5_leaf_vs_parent_mergedt10
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
from tools.analysis.a2_mergedt10_factorial_recompute import (
    GROUPS, N_BOOT, bootstrap, per_user_metrics,
)
from tools.analysis.b2_prototype_variant_overlap import (
    MAIN_DIR_NAME, load_model_dropping_stale_buffers, score_pair,
)

DATASET = "amazon-book"
SEEDS = ["42", "0", "1"]
BASE = ROOT / "log" / "p0" / "taxprocl" / DATASET
LEAF_DIR = BASE / MAIN_DIR_NAME
PARENT_DIR = BASE / "ablation-A5-parent-mergedt10"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "a5_leaf_vs_parent_mergedt10.json")
    args = parser.parse_args(argv)

    t0 = time.time()
    device = torch.device(args.device)
    targets = load_targets(evaluation_protocol_dir(DATASET), "test")
    targets_by_group = {g: targets[g] for g in GROUPS}

    runs = {"leaf": LEAF_DIR, "parent": PARENT_DIR}
    scores = {}
    for name, run_root in runs.items():
        scores[name] = {}
        for seed in SEEDS:
            run_dir = run_root / f"seed{seed}"
            manifest = run_dir / "run_manifest.json"
            if not manifest.is_file():
                raise SystemExit(f"missing checkpoint: {run_dir}")
            if json.loads(manifest.read_text(encoding="utf-8")).get("status") != "completed":
                raise SystemExit(f"run not completed: {run_dir}")
            model, ds, cfg, _ = load_model_dropping_stale_buffers(run_dir, device)
            policy = cfg.get("taxonomy_policy")
            mode = cfg.get("prototype_mode")
            if policy != "merge_t10" or mode != ("leaf" if name == "leaf" else "parent"):
                raise SystemExit(f"{run_dir}: unexpected configuration taxonomy_policy={policy}, prototype_mode={mode}")
            scores[name][seed] = per_user_metrics(model, ds, device, targets_by_group)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[scored] {name} elapsed={time.time()-t0:.0f}s", flush=True)

    out = {"dataset": DATASET, "leaf": str(LEAF_DIR.relative_to(ROOT)), "parent": str(PARENT_DIR.relative_to(ROOT)),
           "n_boot": args.n_boot, "seeds": SEEDS, "point_estimates": {}, "comparison_parent_minus_leaf": {},
           "top20_overlap": []}

    for name in runs:
        out["point_estimates"][name] = {}
        for metric in ("recall", "ndcg"):
            out["point_estimates"][name][metric] = {}
            for g in GROUPS:
                per_seed = [float(np.mean(list(scores[name][s][metric][g].values()))) for s in SEEDS]
                out["point_estimates"][name][metric][g] = {
                    "per_seed": per_seed, "mean": float(np.mean(per_seed)),
                    "std": float(np.std(per_seed, ddof=1))}

    rng = np.random.default_rng(42)
    for metric in ("recall", "ndcg"):
        out["comparison_parent_minus_leaf"][metric] = {}
        for g in GROUPS:
            pooled = {}
            for seed in SEEDS:
                users = set(scores["parent"][seed][metric][g]) & set(scores["leaf"][seed][metric][g])
                for u in users:
                    pooled.setdefault(u, []).append(
                        scores["parent"][seed][metric][g][u] - scores["leaf"][seed][metric][g][u])
            stat = bootstrap([float(np.mean(v)) for v in pooled.values()], args.n_boot, rng)
            leaf_mean = out["point_estimates"]["leaf"][metric][g]["mean"]
            stat["pct_change_vs_leaf"] = 100.0 * (out["point_estimates"]["parent"][metric][g]["mean"] - leaf_mean) / leaf_mean
            out["comparison_parent_minus_leaf"][metric][g] = stat
            print(f"parent-leaf {metric:6s} {g:10s} n={stat['n_users']:6d} diff={stat['mean_diff']:+.6f} "
                  f"({stat['pct_change_vs_leaf']:+.2f}%) CI=[{stat['ci95_lo']:+.6f},{stat['ci95_hi']:+.6f}] "
                  f"excl0={stat['excludes_zero']}", flush=True)

    for seed in SEEDS:
        row = score_pair(PARENT_DIR / f"seed{seed}", LEAF_DIR / f"seed{seed}", device)
        row["seed"] = int(seed)
        out["top20_overlap"].append(row)
        print(json.dumps(row), flush=True)
    for key in ("mean_topk_overlap", "frac_identical_set"):
        out["top20_overlap_mean_" + key] = float(np.mean([r[key] for r in out["top20_overlap"]]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Saved to", args.output, f"elapsed={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
