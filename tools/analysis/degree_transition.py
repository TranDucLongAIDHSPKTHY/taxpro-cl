"""Pool-to-train degree-bucket transition of the evaluated items
(Online Resource 1, Table S21; main paper Section 4.3.1).

For every dataset, takes the items present in the pre-split pool (pool degree
>= 1), assigns each to the bucket of its train-time degree (Strict-Cold 0,
Near-Cold 1-5, Mid-Tail 6-10, Warm >10) and to the bucket of its pool degree, and
counts how many items sit in a sparser bucket at train time than in the pool --
the split-induced reclassification. Items absent from the pool (removed by the
5-core filter, or, for Protocol A, present only in the test file) are not counted.

Pool = train + validation for Protocol A (amazon-book, yelp2018; test excluded)
and train + validation + test for Protocol B (musical-instruments,
arts-crafts-and-sewing), as in the paper. Reads only dataset_verify/<dataset>/
{train,validation,test}.txt.

Usage:
    python -m tools.analysis.degree_transition
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "amazon-book": ("train", "validation"),
    "yelp2018": ("train", "validation"),
    "musical-instruments": ("train", "validation", "test"),
    "arts-crafts-and-sewing": ("train", "validation", "test"),
}
BUCKETS = ("SC", "NC", "MT", "WM")


def bucket(degree: int) -> str:
    if degree == 0:
        return "SC"
    if degree <= 5:
        return "NC"
    if degree <= 10:
        return "MT"
    return "WM"


def read_degrees(dataset: str, split: str):
    degrees = collections.Counter()
    items = set()
    for line in (ROOT / "dataset_verify" / dataset / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        for item in parts[1:]:
            degrees[int(item)] += 1
            items.add(int(item))
    return degrees, items


def analyse(dataset: str):
    per_split = {s: read_degrees(dataset, s) for s in ("train", "validation", "test")}
    train_degree = per_split["train"][0]
    pool_degree = collections.Counter()
    for split in PROTOCOL[dataset]:
        pool_degree.update(per_split[split][0])
    pool_items = set(pool_degree)
    table = {b: {"total": 0, "from_higher": 0, "from": collections.Counter()} for b in BUCKETS}
    order = {b: i for i, b in enumerate(BUCKETS)}
    for item in pool_items:
        train_bucket = bucket(train_degree.get(item, 0))
        pool_bucket = bucket(pool_degree.get(item, 0))
        table[train_bucket]["total"] += 1
        if order[pool_bucket] > order[train_bucket]:
            table[train_bucket]["from_higher"] += 1
            table[train_bucket]["from"][pool_bucket] += 1
    return {
        b: {"total": v["total"], "from_higher": v["from_higher"], "from": dict(v["from"]),
            "rate_percent": 100.0 * v["from_higher"] / v["total"] if v["total"] else None}
        for b, v in table.items() if v["total"]
    }


def main() -> int:
    results = {}
    for dataset in PROTOCOL:
        results[dataset] = analyse(dataset)
        print(f"== {dataset} (pool = {'+'.join(PROTOCOL[dataset])})")
        for b, v in results[dataset].items():
            detail = " + ".join(f"{n} from {k}" for k, n in sorted(v["from"].items()))
            rate = "n/a" if v["rate_percent"] is None else f"{v['rate_percent']:.2f}%"
            print(f"   {b}: total {v['total']:6d}  from higher pool bucket {v['from_higher']:5d} ({detail or '-'})  {rate}")
    out = ROOT / "results" / "degree_transition.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Saved to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
