"""Screening criteria S1-S3 of docs/confirmatory_protocol.md for Amazon 2018 5-core files.

Runs the Protocol B pipeline of tools/data/build_arts_crafts_and_sewing.py in memory
(deduplicate, remap, iterative 5-core, per-user 70/10/20 split with seed 42) and
reports, from data statistics only:
  S1  interactions after preprocessing,
  S2  Near-Cold-eligible test users (a test positive whose train degree is 1-5),
  S3  Long-Tail-eligible test users (train degree 1-10).
Nothing is written to dataset/ or dataset_verify/, and no model is involved.

Usage:
    python -m tools.data.screen_confirmatory_candidates <Category>_5.json.gz [...] --out screening.json
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data.build_arts_crafts_and_sewing import (  # noqa: E402
    MINIMUM_DEGREE, RATIOS, SPLIT_SEED, build_id_maps, per_user_split)
from tools.data.build_splits import iterative_k_core, sha256_file  # noqa: E402


def read_pairs(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    pairs = set()
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                record = json.loads(line)
                pairs.add((record["reviewerID"], record["asin"]))
    return pairs


def screen(path: Path) -> dict:
    org_pairs = read_pairs(path)
    user_map, item_map = build_id_maps(org_pairs)
    remapped = {(user_map[u], item_map[i]) for u, i in org_pairs}
    five_core, _ = iterative_k_core(remapped, MINIMUM_DEGREE)
    train, _validation, test = per_user_split(five_core, SPLIT_SEED, RATIOS)
    degree = collections.Counter(item for _, item in train)
    nc_users = {u for u, i in test if 1 <= degree.get(i, 0) <= 5}
    lt_users = {u for u, i in test if 1 <= degree.get(i, 0) <= 10}
    return {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "distinct_pairs": len(org_pairs),
        "S1_interactions": len(five_core),
        "users": len({u for u, _ in five_core}),
        "items": len({i for _, i in five_core}),
        "S2_near_cold_test_users": len(nc_users),
        "S3_long_tail_test_users": len(lt_users),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    rows = []
    for path in args.files:
        row = screen(path)
        rows.append(row)
        print(json.dumps(row))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
