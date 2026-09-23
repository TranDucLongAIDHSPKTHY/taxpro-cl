"""Leaf-size distribution of each dataset's selected taxonomy policy
(Online Resource 1, Table S15; main paper Section 4.1 and Table 3).

For every dataset, reads metadata/taxonomy_variants/<dataset>/<policy>/item_to_leaf.json
(policy = the one selected for the paper, tools.experiments.run_taxpro.PAPER_TAXONOMY_POLICY),
counts the valid-taxonomy items of each leaf, and reports the number of leaves, the
median leaf size, the number and share of leaves with 1-2 items, and the share of
valid-taxonomy items that sit in those leaves.

Inference-free (reads shipped files only).

Usage:
    python -m tools.analysis.leaf_size_distribution
"""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing")
POLICY = {
    "amazon-book": "merge_t10",
    "yelp2018": "no_merge",
    "musical-instruments": "no_merge",
    "arts-crafts-and-sewing": "no_merge",
}


def leaf_sizes(dataset: str, policy: str):
    path = ROOT / "metadata" / "taxonomy_variants" / dataset / policy / "item_to_leaf.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    sizes = collections.Counter(
        entry["leaf_id"] for entry in mapping.values() if entry.get("valid_taxonomy")
    )
    return len(mapping), sizes


def main() -> int:
    results = []
    for dataset in DATASETS:
        catalog, sizes = leaf_sizes(dataset, POLICY[dataset])
        values = sorted(sizes.values())
        small = [s for s in values if s <= 2]
        valid = sum(values)
        row = {
            "dataset": dataset,
            "policy": POLICY[dataset],
            "total_items_catalog": catalog,
            "total_valid_items": valid,
            "n_leaves": len(values),
            "median_leaf_size": statistics.median(values),
            "n_leaves_1_2": len(small),
            "pct_leaves_1_2": 100.0 * len(small) / len(values),
            "items_in_leaves_1_2": sum(small),
            "pct_items_in_leaves_1_2": 100.0 * sum(small) / valid,
        }
        results.append(row)
        print(f"{dataset:24s} leaves={row['n_leaves']:5d} median={row['median_leaf_size']:6.1f} "
              f"1-2 item leaves={row['n_leaves_1_2']}/{row['n_leaves']} ({row['pct_leaves_1_2']:.2f}%) "
              f"items in them={row['items_in_leaves_1_2']} ({row['pct_items_in_leaves_1_2']:.2f}% of valid items)")
    out = ROOT / "results" / "leaf_size_distribution.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Saved to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
