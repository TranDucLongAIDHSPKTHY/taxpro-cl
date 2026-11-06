"""Build re-drawn splits of the locked 5-core pools (docs/extended_evaluation_protocol.md).

The pool of a Protocol B dataset is the union of its locked train, validation and test
files. It is partitioned again per user, 70/10/20, with the splitter that built the
locked split, under a new split seed. User and item identifiers are unchanged. Every
re-drawn split gets its own dataset identifier, `<dataset>-split<seed>`, with its own
`dataset_verify/`, evaluation protocol and `no_merge` taxonomy variant, so locked
artifacts are read and never written. The tool stops if the splitter does not reproduce
the locked files under the published split seed 42, or if a locked file changes.

Usage:
    python -m tools.data.build_redrawn_splits [--force]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data.build_office_products import RATIOS, per_user_split  # noqa: E402
from tools.data.build_splits import (  # noqa: E402
    _write_interactions_atomic,
    _write_json_atomic,
    sha256_file,
)
from tools.protocol.build import (  # noqa: E402
    build_evaluation_protocol,
    build_group_masks,
    finalize_taxonomy_variant,
    item_degrees,
    parse_split,
    relative,
)
from utility.utility_data.taxonomy import (  # noqa: E402
    REDRAWN_SPLIT_BASES,
    REDRAWN_SPLIT_SEEDS,
)

LOCKED_SPLIT_SEED = 42
POLICY = "no_merge"
SPLIT_NAMES = ("train", "validation", "test")
ALGORITHM_ID = "taxprocl-redrawn-per-user-shuffle-70-10-20-v1"
COMMAND = "python -m tools.data.build_redrawn_splits"


def redrawn_id(base, seed):
    return "{}-split{}".format(base, int(seed))


def read_pairs(path):
    pairs = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            tokens = [int(token) for token in line.split()]
            if tokens:
                pairs.update((tokens[0], item) for item in tokens[1:])
    return pairs


def locked_parts(base):
    directory = ROOT / "dataset_verify" / base
    return {name: read_pairs(directory / (name + ".txt")) for name in SPLIT_NAMES}


def check_splitter(base, parts):
    """The splitter must rebuild the locked partition from the pool under seed 42."""
    pool = parts["train"] | parts["validation"] | parts["test"]
    if len(pool) != sum(len(part) for part in parts.values()):
        raise AssertionError("{}: locked splits overlap".format(base))
    rebuilt = per_user_split(pool, LOCKED_SPLIT_SEED, RATIOS)
    for name, part in zip(SPLIT_NAMES, rebuilt):
        if part != parts[name]:
            raise AssertionError("{}: splitter does not reproduce locked {}".format(base, name))
    return pool


def locked_hashes(base):
    verify = ROOT / "dataset_verify" / base
    variant = ROOT / "metadata" / "taxonomy_variants" / base / POLICY
    paths = [verify / (name + ".txt") for name in SPLIT_NAMES]
    paths += [variant / "manifest.json", variant / "mapping_provenance.jsonl"]
    return {relative(path): sha256_file(path) for path in paths}


def write_split(dataset_id, base, seed, pool):
    directory = ROOT / "dataset_verify" / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    train, validation, test = per_user_split(pool, seed, RATIOS)
    if train & validation or train & test or validation & test:
        raise AssertionError("Split partitions are not disjoint")
    if train | validation | test != pool:
        raise AssertionError("Split partition does not cover the pool")
    for name, part in zip(SPLIT_NAMES, (train, validation, test)):
        _write_interactions_atomic(directory / (name + ".txt"), part)
    manifest = {
        "schema_version": 1,
        "algorithm": ALGORITHM_ID,
        "dataset": dataset_id,
        "base_dataset": base,
        "split_seed": int(seed),
        "ratios": RATIOS,
        "pool_interactions": len(pool),
        "source_hashes": {
            name: sha256_file(ROOT / "dataset_verify" / base / (name + ".txt"))
            for name in SPLIT_NAMES
        },
        "output_hashes": {
            name: sha256_file(directory / (name + ".txt")) for name in SPLIT_NAMES
        },
    }
    _write_json_atomic(directory / "split_manifest.json", manifest)
    return manifest


def group_shares(degrees):
    """Percentage of training-observed items in the NC and LT groups."""
    observed = [degree for degree in degrees if degree > 0]
    return {
        "near_cold_percent": round(100.0 * sum(degree <= 5 for degree in observed) / len(observed), 3),
        "long_tail_percent": round(100.0 * sum(degree <= 10 for degree in observed) / len(observed), 3),
    }


def build_protocol_and_taxonomy(dataset_id, base, seed):
    verify = ROOT / "dataset_verify" / dataset_id
    variant = ROOT / "metadata" / "taxonomy_variants" / base / POLICY
    provenance_path = variant / "mapping_provenance.jsonl"
    with provenance_path.open("r", encoding="utf-8") as stream:
        base_rows = [json.loads(line) for line in stream if line.strip()]
    catalog_items = len(base_rows)
    if [row["item_id"] for row in base_rows] != list(range(catalog_items)):
        raise AssertionError("{}: provenance is not indexed by item id".format(base))

    splits = {
        name: parse_split(verify / (name + ".txt"), catalog_items) for name in SPLIT_NAMES
    }
    degrees = item_degrees(splits["train"]["pairs"], catalog_items)
    masks = build_group_masks(degrees)
    shares = group_shares(degrees)
    command = COMMAND + "  # dataset {}".format(dataset_id)
    build_evaluation_protocol(
        dataset_id, degrees, masks, splits["validation"], splits["test"], splits,
        split_seed=int(seed), creation_command=command,
    )

    # Under no_merge the label of an item does not depend on the split; only train
    # support and degree do.
    support = collections.Counter(
        row["original_leaf"]
        for row, degree in zip(base_rows, degrees)
        if degree > 0 and row["original_leaf"] is not None
    )
    provenance = []
    for row, degree in zip(base_rows, degrees):
        updated = dict(row)
        updated["train_degree"] = degree
        updated["original_train_support"] = support[row["original_leaf"]] if row["original_leaf"] else 0
        updated["final_train_support"] = support[row["final_leaf"]] if row["final_leaf"] else 0
        provenance.append(updated)
    with (variant / "taxonomy_statistics.json").open("r", encoding="utf-8") as stream:
        base_statistics = json.load(stream)
    statistics, _ = finalize_taxonomy_variant(
        dataset_id,
        POLICY,
        None,
        [row["final_leaf"] for row in base_rows],
        provenance,
        base_statistics["metadata_missing_items"],
        sum(degree > 0 for degree in degrees),
        {"train": relative(verify / "train.txt"), "base_provenance": relative(provenance_path)},
        {"train": sha256_file(verify / "train.txt"), "base_provenance": sha256_file(provenance_path)},
        base_statistics["policy"],
        creation_command=command,
    )
    return {**statistics, **shares}


def build_all(force=False):
    report = {}
    for base in REDRAWN_SPLIT_BASES:
        before = locked_hashes(base)
        parts = locked_parts(base)
        pool = check_splitter(base, parts)
        locked_degrees = collections.Counter(item for _, item in parts["train"])
        locked = group_shares(list(locked_degrees.values()))
        for seed in REDRAWN_SPLIT_SEEDS:
            dataset_id = redrawn_id(base, seed)
            if (ROOT / "dataset_verify" / dataset_id / "split_manifest.json").exists() and not force:
                raise FileExistsError("{} exists; use --force to rebuild it".format(dataset_id))
            manifest = write_split(dataset_id, base, seed, pool)
            statistics = build_protocol_and_taxonomy(dataset_id, base, seed)
            report[dataset_id] = {
                "output_hashes": manifest["output_hashes"],
                "train_observed_items": statistics["train_observed_items"],
                "final_leaf_count": statistics["final_leaf_count"],
                "coverage_percent": statistics["coverage_percent"],
                "near_cold_percent": statistics["near_cold_percent"],
                "long_tail_percent": statistics["long_tail_percent"],
                "locked_near_cold_percent": locked["near_cold_percent"],
                "locked_long_tail_percent": locked["long_tail_percent"],
            }
            print("{}: items={} leaves={} NC {} (locked {}) LT {} (locked {})".format(
                dataset_id, statistics["train_observed_items"], statistics["final_leaf_count"],
                statistics["near_cold_percent"], locked["near_cold_percent"],
                statistics["long_tail_percent"], locked["long_tail_percent"]))
        if locked_hashes(base) != before:
            raise AssertionError("{}: a locked artifact changed".format(base))
    output = ROOT / "results" / "extended" / "redrawn_split_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing re-drawn splits.")
    args = parser.parse_args(argv)
    build_all(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
