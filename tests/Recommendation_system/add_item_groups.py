"""One-time backfill: add near_cold/long_tail/warm item-group columns to an
existing result/ranking_details.csv that predates them, and regenerate
result/ranking_summary.csv (now broken down by Item_Group) from the
backfilled details.

Pure post-processing -- no model inference. An item's group only depends on
its (Dataset, Item) train-degree (preprocessed/evaluation_protocol/
split_seed_42/<dataset>/item_degrees_train.json), which is independent of
which model produced the rank, so this never needs to touch a checkpoint.

Run once, from the repo root, after a sweep finishes:
    python -m tests.Recommendation_system.add_item_groups
"""

from __future__ import annotations

import csv
import sys
from array import array
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.Recommendation_system import csv_store, rank_comparison

RESULT_DIR = Path(__file__).resolve().parent / "result"
DETAILS_CSV = RESULT_DIR / "ranking_details.csv"
SUMMARY_CSV = RESULT_DIR / "ranking_summary.csv"
MANIFEST_CSV = RESULT_DIR / "checkpoint_manifest.csv"

_TRUE_STRINGS = {"True", "true", "1"}


def _load_manifest_lookup():
    lookup = {}
    if not MANIFEST_CSV.exists():
        return lookup
    for row in csv_store.stream_rows(MANIFEST_CSV):
        lookup[(row["Dataset"], row["Model"])] = (
            row["Checkpoint_Dir"], float(row["Validation_Score"]),
        )
    return lookup


def _new_bucket():
    return {
        "delta_ranks": array("i"), "in_b": array("B"), "in_t": array("B"),
        "near_cold": array("B"), "long_tail": array("B"), "warm": array("B"),
        "user": array("l"),
    }


def backfill():
    if not DETAILS_CSV.exists():
        raise FileNotFoundError(DETAILS_CSV)
    manifest_lookup = _load_manifest_lookup()

    degrees_cache = {}

    def degrees_for(dataset_name):
        if dataset_name not in degrees_cache:
            degrees_cache[dataset_name] = rank_comparison.load_item_degrees(dataset_name)
        return degrees_cache[dataset_name]

    group_state = defaultdict(_new_bucket)
    tmp_details = DETAILS_CSV.with_name(DETAILS_CSV.name + ".backfill_tmp")

    n = 0
    with DETAILS_CSV.open("r", encoding="utf-8", newline="") as in_stream, \
         tmp_details.open("w", encoding="utf-8", newline="") as out_stream:
        reader = csv.DictReader(in_stream)
        if "Item_Degree" in (reader.fieldnames or []):
            print("ranking_details.csv already has item-group columns; nothing to backfill.")
            return
        writer = csv.DictWriter(out_stream, fieldnames=rank_comparison.DETAIL_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            dataset_name = row["Dataset"]
            item = int(row["Item"])
            degree = int(degrees_for(dataset_name)[item])
            near_cold = 1 <= degree <= rank_comparison.NEAR_COLD_MAX_DEGREE
            long_tail = 1 <= degree <= rank_comparison.LONG_TAIL_MAX_DEGREE
            warm = degree > rank_comparison.LONG_TAIL_MAX_DEGREE

            row["Item_Degree"] = degree
            row["Is_Near_Cold"] = near_cold
            row["Is_Long_Tail"] = long_tail
            row["Is_Warm"] = warm
            writer.writerow({key: row.get(key, "") for key in rank_comparison.DETAIL_FIELDNAMES})

            k = int(row["K"])
            acc = group_state[(dataset_name, row["Baseline"], k)]
            acc["delta_ranks"].append(int(row["Delta_Rank"]))
            acc["in_b"].append(1 if row["In_TopK_Baseline"] in _TRUE_STRINGS else 0)
            acc["in_t"].append(1 if row["In_TopK_TaxProCL"] in _TRUE_STRINGS else 0)
            acc["near_cold"].append(1 if near_cold else 0)
            acc["long_tail"].append(1 if long_tail else 0)
            acc["warm"].append(1 if warm else 0)
            acc["user"].append(int(row["User"]))

            n += 1
            if n % 2_000_000 == 0:
                print("...", n, "rows processed")

    tmp_details.replace(DETAILS_CSV)
    print("Backfilled item-group columns into {} ({} rows)".format(DETAILS_CSV, n))

    summary_rows = []
    for (dataset_name, baseline_name, k), acc in sorted(group_state.items()):
        checkpoint_baseline, baseline_val = manifest_lookup.get((dataset_name, baseline_name), ("", 0.0))
        checkpoint_taxpro, taxpro_val = manifest_lookup.get((dataset_name, "TaxPro-CL"), ("", 0.0))
        summary_rows.extend(rank_comparison.build_group_summary_rows(
            dataset_name, baseline_name, k,
            acc["delta_ranks"], acc["in_b"], acc["in_t"],
            acc["near_cold"], acc["long_tail"], acc["warm"], acc["user"],
            checkpoint_baseline, checkpoint_taxpro, baseline_val, taxpro_val,
        ))

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as out_stream:
        writer = csv.DictWriter(out_stream, fieldnames=rank_comparison.SUMMARY_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    print("Rewrote {} with {} rows (Item_Group breakdown)".format(SUMMARY_CSV, len(summary_rows)))


if __name__ == "__main__":
    backfill()
