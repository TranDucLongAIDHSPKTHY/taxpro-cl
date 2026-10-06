"""Composition of the train-degree item groups on the test split (Online Resource 1, Table S21b): for every dataset
and group (NC, MT, LT, Wm), the number of items, their share of the items with at least one training interaction, the
number and share of test positives in the group, and the number of eligible test users (users with at least one test
positive in the group). Reads the shipped evaluation protocol only. Usage: python -m tools.analysis.group_composition
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir  # noqa: E402
from utility.utility_train.group_evaluator import load_targets  # noqa: E402

DATASETS = ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing", "office-products")


def main():
    res = {}
    for ds in DATASETS:
        pdir = evaluation_protocol_dir(ds)
        m = np.load(Path(pdir) / "group_masks.npz")
        overall = load_targets(pdir, "test")["overall"]
        total = sum(len(v) for v in overall.values())
        trained = m["long_tail"] | m["warm"]
        groups = {"NC": m["near_cold"], "MT": m["long_tail"] & ~m["near_cold"], "LT": m["long_tail"], "Wm": m["warm"]}
        row = {}
        for g, mask in groups.items():
            pos = sum(sum(1 for i in v if mask[i]) for v in overall.values())
            row[g] = {"items": int(mask.sum()), "item_share_of_trained": float(mask.sum() / trained.sum()),
                      "test_positives": int(pos), "positive_share": pos / total,
                      "eligible_users": sum(1 for v in overall.values() if any(mask[i] for i in v))}
        res[ds] = {"items_with_train_interactions": int(trained.sum()), "test_positives": int(total),
                   "test_users": len(overall), "groups": row}
    out = ROOT / "results" / "group_composition.json"
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
