"""Environment check for the Amazon-Book factorial (Online Resource 1, Table S23b;
main paper Section 5.4).

V0-V2 (merge_t10) ran on Environment A, V3 (A2-V3-warm20) on Environment B.
This replaces V3 with the Environment-A main checkpoint (TaxPro-CL-main) and
repeats, with exactly the functions of tools/analysis/factorial_multiplicity.py,
the four conditional contrasts and the interaction (V3-V2)-(V1-V0) -- per-user
bootstrap CI and paired sign-flip p (1e5 permutations, seed 42; 1e6, seed 123
for the primary groups) -- then re-runs the four 8-cell Holm families with the
Amazon-Book cells replaced and the other datasets' cells taken from
results/factorial_multiplicity.json.

Inference only, no retraining.

Usage:
    python -m tools.analysis.factorial_v3main_substitution
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

import tools.analysis.factorial_multiplicity as base  # noqa: E402

MAIN_V3 = ("log/p0/taxprocl/amazon-book/taxpro-cl-v15-prototype-leaf-lambda0.5-same_leaf_weight0-"
           "user_ssl-warmstart20-noblend-temp0.1-DONE-overall-2.30pct-BEST-overall-nearcold-longtail-positive")
PAIRS = {"direction": ("direction_fixed_eps__V1-V0", "direction_adaptive_eps__V3-V2"),
         "epsilon": ("eps_random_dir__V2-V0", "eps_taxonomy_dir__V3-V1")}
OTHER_DATASETS = ("yelp2018", "musical-instruments", "arts-crafts-and-sewing")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "factorial_v3main_substitution.json")
    args = parser.parse_args(argv)
    t0 = time.time()
    rel_dirs = {**base.CHECKPOINT_SETS["amazon-book"]["current"], "V3": MAIN_V3}
    scores = base.score_checkpoint_set("amazon-book", rel_dirs, torch.device(args.device))
    comparisons = dict(base.COMPARISONS)
    comparisons["interaction__(V3-V2)-(V1-V0)"] = [(+1, "V3"), (-1, "V2"), (-1, "V1"), (+1, "V0")]
    cells = {}
    for name, terms in comparisons.items():
        for metric in ("recall", "ndcg"):
            for group in base.GROUPS:
                diffs = list(base.pooled_diffs(scores, terms, metric, group).values())
                ci = base.bootstrap_ci(diffs, base.N_BOOT, np.random.default_rng(42))
                p5 = base.sign_flip_pvalue(diffs, base.N_PERM_PRIMARY, np.random.default_rng(42))
                p6 = (base.sign_flip_pvalue(diffs, base.N_PERM_PRECISION, np.random.default_rng(base.PRECISION_SEED))
                      if group in base.HOLM_GROUPS else None)
                cells[f"{name}|{metric}|{group}"] = {**ci, "p1e5": p5, "p1e6": p6}
                print(name, metric, group, f"diff={ci['mean_diff']:+.3e} p={p5:.5f}", flush=True)
    original = json.loads((ROOT / "results" / "factorial_multiplicity.json").read_text(encoding="utf-8"))
    holm = {}
    for metric in ("recall", "ndcg"):
        for factor, (a, b) in PAIRS.items():
            for key, orig_key in (("p1e5", "p_signflip_1e5_seed42"), ("p1e6", "p_signflip_1e6_seed123")):
                family = [{"dataset": "amazon-book", "group": g,
                           "p": max(cells[f"{a}|{metric}|{g}"][key], cells[f"{b}|{metric}|{g}"][key])}
                          for g in base.HOLM_GROUPS]
                family += [{"dataset": ds, "group": g,
                            "p": original["combined_cells"][f"{ds}|current|{factor}|{metric}"][g][orig_key]}
                           for ds in OTHER_DATASETS for g in base.HOLM_GROUPS]
                holm[f"{factor}|{metric}|{key}"] = base.holm_bonferroni(family, "p", "sig")
    args.output.write_text(json.dumps({"v3": MAIN_V3, "cells": cells, "holm": holm}, indent=1) + "\n",
                           encoding="utf-8")
    print("Saved to", args.output, f"elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
