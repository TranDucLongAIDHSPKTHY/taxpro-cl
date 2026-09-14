"""Print TaxPro-CL's actual Top-K recommendation list for a user (exactly K
items, ranked 1..K by TaxPro-CL), with each item's baseline rank alongside
for comparison -- e.g.

 STT  Rank_TaxProCL       Item  Rank_SimGCL  Delta  Group
   1              1        687            4     -3  warm
   2              2        775            3     +1  warm
   3              3   721 (GT)            2     -1  warm

Items with "(GT)" after the item id are ground-truth: the user actually
interacted with them in the held-out test set (the model never sees this at
scoring time -- see how full-catalog ranking works in inference.py). A large
Rank_<baseline> value (e.g. in the thousands) means the baseline buried that
item deep in the catalog while TaxPro-CL surfaced it into the Top-K.

Use --against-baseline to flip this around and print the baseline's own
Top-K list instead, with TaxPro-CL's rank alongside.

Reads result/ranking_details.csv, which already holds -- for the requested
(dataset, baseline, K) -- every item in the union of both models' Top-K plus
every ground-truth positive for every user, so no new scoring is needed.

Usage (from the repository root):
    python -m tests.Recommendation_system.show_user_ranking \
        --dataset amazon-book --baseline SimGCL --user 1 --k 20 [--against-baseline]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_DIR = Path(__file__).resolve().parent / "result"
DETAILS_CSV = RESULT_DIR / "ranking_details.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--against-baseline", action="store_true",
        help="Show the baseline's own Top-K list instead of TaxPro-CL's.",
    )
    return parser.parse_args(argv)


def _group_label(row):
    if row["Is_Near_Cold"] == "True":
        return "near_cold"
    if row["Is_Long_Tail"] == "True":
        return "long_tail"
    return "warm"


def main(argv=None):
    args = parse_args(argv)
    k_str = str(args.k)

    if not DETAILS_CSV.exists():
        raise FileNotFoundError("{} not found -- run run_comparison.py first".format(DETAILS_CSV))

    primary_key = "Rank_Baseline" if args.against_baseline else "Rank_TaxProCL"
    other_key = "Rank_TaxProCL" if args.against_baseline else "Rank_Baseline"

    rows = []
    with DETAILS_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if (
                row["Dataset"] == args.dataset
                and row["Baseline"] == args.baseline
                and row["K"] == k_str
                and row["User"] == args.user
                and int(row[primary_key]) <= args.k
            ):
                rows.append(row)

    if not rows:
        print("No rows found for dataset={} baseline={} user={} K={}. "
              "Check the ids exist and run_comparison.py has been run for this pair."
              .format(args.dataset, args.baseline, args.user, args.k))
        return 1

    primary_col = "Rank_" + args.baseline if args.against_baseline else "Rank_TaxProCL"
    other_col = "Rank_TaxProCL" if args.against_baseline else "Rank_" + args.baseline
    owner = args.baseline if args.against_baseline else "TaxPro-CL"
    print("Top-{} theo {} (dataset={}, user={})".format(args.k, owner, args.dataset, args.user))
    print("{:>4} {:>14} {:>10} {:>14}  {:>6}  {}".format(
        "STT", primary_col, "Item", other_col, "Delta", "Group"))
    for stt, row in enumerate(sorted(rows, key=lambda r: int(r[primary_key])), start=1):
        item_label = row["Item"] + (" (GT)" if row["Is_Ground_Truth"] == "True" else "")
        delta = int(row["Delta_Rank"])
        print("{:>4} {:>14} {:>10} {:>14}  {:>+6}  {}".format(
            stt, row[primary_key], item_label, row[other_key], delta, _group_label(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
