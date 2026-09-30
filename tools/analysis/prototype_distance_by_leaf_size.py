"""Relative distance between a leaf prototype and its items' layer-0 embeddings, by leaf size.

Superseded for Online Resource 1, Table S16 by
tools/analysis/prototype_distance_by_space.py: the layer-0 lookup used here
belongs neither to the space the prototype bank is built from (layer-averaged
propagated embedding) nor to the space Eq. (1) uses (per-layer embedding), so
its distances mix embedding spaces. Kept for the "Layer 0 (lookup)" row of S16.

For each dataset's main TaxPro-CL checkpoint (all three training seeds, or the one
given with --seed), computes for every
valid-taxonomy item i of leaf l the ratio ||p_l - e_i|| / ||e_i||, where p_l is the
trained EMA prototype (model.prototype_bank.prototypes) and e_i the item's ego
(layer-0 lookup) embedding, groups the items by the number of valid items in their
leaf (1, 2, 3-10, 11-100, 101+) and reports the median per group.

Inference only; needs the trained checkpoint, which is selected exactly as by the
other analysis scripts (tools.ranking.checkpoint_selection).

Usage:
    python -m tools.analysis.prototype_distance_by_leaf_size [--seed 42] [--device cpu]

Table S16 of Online Resource 1 reports three-seed means; the "max_seed_spread"
field gives the largest seed-to-seed spread of this script's medians.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ranking import checkpoint_selection  # noqa: E402
from tools.analysis.prototype_variants_overlap import load_model_dropping_stale_buffers  # noqa: E402

DATASETS = ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing")
BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3-10", 3, 10), ("11-100", 11, 100), ("101+", 101, 10 ** 9))


def analyse(dataset: str, seed: str, device: torch.device):
    choice = checkpoint_selection.select_checkpoint("TaxPro-CL", dataset)
    run_dir = choice.run_dir.parent / f"seed{seed}"
    model, _data, _config, _name = load_model_dropping_stale_buffers(run_dir, device)
    model.eval()
    with torch.no_grad():
        ego = model.item_embedding.weight
        prototypes = model.prototype_bank.prototypes
        leaf = model.item_to_leaf_id.long()
        valid = model.valid_train_mask.bool()
        sizes = collections.Counter(leaf[valid].tolist())
        size_of_item = torch.tensor(
            [sizes.get(int(l), 0) if v else 0 for l, v in zip(leaf.tolist(), valid.tolist())]
        )
        ratio = (prototypes[leaf.clamp_min(0)] - ego).norm(dim=1) / ego.norm(dim=1).clamp_min(1e-12)
        rows = {}
        for name, lo, hi in BUCKETS:
            mask = valid & (size_of_item >= lo) & (size_of_item <= hi)
            rows[name] = {
                "n_items": int(mask.sum()),
                "median_relative_distance": float(ratio[mask].median()) if mask.any() else None,
            }
    return str(run_dir.relative_to(ROOT)).replace("\\", "/"), rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, help="a single seed (default: 42, 0 and 1)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    seeds = [args.seed] if args.seed else ["42", "0", "1"]
    results = {}
    for dataset in DATASETS:
        per_seed = {}
        for seed in seeds:
            run_dir, rows = analyse(dataset, seed, device)
            per_seed[seed] = {"run_dir": run_dir, "by_leaf_size": rows}
            cells = "  ".join(
                f"{name}: {row['median_relative_distance']:.3f} (n={row['n_items']})"
                if row["median_relative_distance"] is not None else f"{name}: n/a"
                for name, row in rows.items()
            )
            print(f"{dataset:24s} seed {seed:>2s}  {cells}")
        spread = 0.0
        for name, _lo, _hi in BUCKETS:
            values = [s["by_leaf_size"][name]["median_relative_distance"] for s in per_seed.values()]
            values = [v for v in values if v is not None]
            if len(values) > 1:
                spread = max(spread, max(values) - min(values))
        results[dataset] = {"per_seed": per_seed, "max_seed_spread": spread}
    out = ROOT / "results" / "prototype_distance_by_leaf_size.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Saved to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
