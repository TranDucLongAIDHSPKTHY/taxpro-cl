"""Beyond-accuracy diagnostics for the six-method comparison.

TaxPro-CL trades lower Overall/Warm Recall@20 for higher Near-Cold/Long-Tail
(Section 5.1 of the manuscript). This computes three metrics that let a
reader judge whether that trade buys anything, from the SAME completed
checkpoints already used for the main results table -- no retraining:

- catalog coverage@20: fraction of the full item catalog that appears in at
  least one user's top-20 list.
- long_tail_share@20: fraction of (user, top-20 slot) pairs whose item has
  train degree in [1, 10] (this paper's Long-Tail definition).
- avg_recommended_popularity: mean train degree of recommended items,
  averaged over every (user, slot) pair -- lower means less popularity bias.

Also reports the harmonic mean of Long-Tail and Overall Recall@20 (already
computed, read from final_test_group_metrics.json) as a single
balance-of-trade-off number per method.

Reuses tests/Recommendation_system/inference.py (checkpoint loading, full-
catalog ranking) and checkpoint_selection.py (same official-checkpoint choice
used everywhere else), so results are directly comparable to the main table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.Recommendation_system.checkpoint_selection import select_checkpoint
from tests.Recommendation_system.inference import compute_batch_order_and_rank, load_model

MODELS = ("LightGCN", "SGL", "SimGCL", "XSimGCL", "NCL", "TaxPro-CL")
DATASETS = ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing")
LONG_TAIL_MAX_DEGREE = 10
K = 20
BATCH_SIZE = 512


def load_item_degrees(dataset_name):
    path = (
        ROOT
        / "preprocessed"
        / "evaluation_protocol"
        / "split_seed_42"
        / dataset_name
        / "item_degrees_train.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["degrees"]


def load_group_recall(run_dir):
    payload = json.loads(
        (Path(run_dir) / "final_test_group_metrics.json").read_text(encoding="utf-8")
    )
    return {
        "long_tail_recall20": payload["long_tail"]["recall"]["20"],
        "overall_recall20": payload["overall"]["recall"]["20"],
    }


def harmonic_mean(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def compute_for_model_dataset(model_name, dataset_name, device):
    choice = select_checkpoint(model_name, dataset_name)
    model, dataset, _config, _name = load_model(choice.run_dir, device)
    degrees = load_item_degrees(dataset_name)

    recommended_items = set()
    long_tail_slots = 0
    total_slots = 0
    degree_sum = 0
    all_users = list(range(dataset.num_users))
    for start in range(0, len(all_users), BATCH_SIZE):
        batch = all_users[start : start + BATCH_SIZE]
        order, _rank = compute_batch_order_and_rank(model, dataset, device, batch, split="test")
        for row in order[:, :K].tolist():
            for item_id in row:
                degree = degrees[item_id] if item_id < len(degrees) else 0
                recommended_items.add(item_id)
                total_slots += 1
                degree_sum += degree
                if 1 <= degree <= LONG_TAIL_MAX_DEGREE:
                    long_tail_slots += 1

    recall = load_group_recall(choice.run_dir)
    return {
        "model": model_name,
        "dataset": dataset_name,
        "checkpoint": str(choice.run_dir.relative_to(ROOT)),
        "seed": choice.seed,
        "coverage_at_20": len(recommended_items) / dataset.num_items,
        "long_tail_share_at_20": long_tail_slots / total_slots if total_slots else 0.0,
        "avg_recommended_popularity": degree_sum / total_slots if total_slots else 0.0,
        "long_tail_recall20": recall["long_tail_recall20"],
        "overall_recall20": recall["overall_recall20"],
        "harmonic_mean_longtail_overall": harmonic_mean(
            recall["long_tail_recall20"], recall["overall_recall20"]
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "beyond_accuracy.json"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows = []
    for dataset_name in args.datasets:
        for model_name in args.models:
            try:
                row = compute_for_model_dataset(model_name, dataset_name, device)
            except FileNotFoundError as error:
                print("SKIP {}/{}: {}".format(model_name, dataset_name, error), file=sys.stderr)
                continue
            rows.append(row)
            print(json.dumps(row, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Saved {} rows to {}".format(len(rows), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
