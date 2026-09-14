"""Build taxonomy variants + evaluation protocol for arts-crafts-and-sewing.

Additive companion to tools/protocol/build.py: that script's DATASETS dict
and build_amazon_variants() hardcode "amazon-book" internally (dataset_name
is not a parameter, and the root-category check is literally
`if path == ["Books"]`), so it cannot process a second Amazon-family
dataset without being edited. This file avoids touching it: it imports the
already dataset_name-parameterized pieces (finalize_taxonomy_variant,
build_evaluation_protocol, parse_split, item_degrees, build_group_masks,
choose_amazon_ancestor, taxonomy_record, distribution, THRESHOLDS,
VARIANT_NAMES, GROUP_DEFINITIONS) and re-implements only
build_amazon_variants' loop body, parameterized by dataset_name and
root_category instead of the hardcoded "amazon-book"/"Books". Zero lines
of tools/protocol/build.py are modified.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.protocol.build import (  # noqa: E402  (reused, not duplicated)
    GROUP_DEFINITIONS,
    THRESHOLDS,
    VARIANT_NAMES,
    build_evaluation_protocol,
    build_group_masks,
    choose_amazon_ancestor,
    distribution,
    finalize_taxonomy_variant,
    item_degrees,
    parse_split,
    relative,
    sha256_file,
    taxonomy_record,
)

DATASET_NAME = "arts-crafts-and-sewing"
ROOT_CATEGORY = "Arts, Crafts & Sewing"
CATALOG_ITEMS = None  # set from item_list.txt at runtime, see main()
VERIFY_DIR = ROOT / "dataset_verify" / DATASET_NAME
MAPPING_PATH = ROOT / "metadata" / "item2category_arts_crafts_and_sewing.json"
MERGED_PATH = ROOT / "metadata" / "merged_metadata_arts_crafts_and_sewing.json"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def count_catalog_items():
    item_list_path = ROOT / "dataset" / DATASET_NAME / "item_list.txt"
    with item_list_path.open("r", encoding="utf-8") as stream:
        lines = [line for line in stream.read().splitlines() if line.strip()]
    return len(lines) - 1  # minus header row


def build_arts_crafts_and_sewing_taxonomy_variants(degrees, catalog_items):
    """Same algorithm as tools.protocol.build.build_amazon_variants,
    parameterized by dataset_name/root_category instead of hardcoded."""
    mapping = load_json(MAPPING_PATH)
    merged = load_json(MERGED_PATH)
    original_paths = []
    metadata_missing = 0
    for item in range(catalog_items):
        record, path = taxonomy_record(item, mapping)
        merged_record = merged.get(str(item))
        merged_paths = [
            candidate
            for candidate in (merged_record or {}).get("taxonomy_paths") or []
            if candidate
        ]
        if not path and merged_paths:
            path = merged_paths[0]
        missing = (
            not path
            or not merged_record
            or merged_record.get("status") == "missing"
        )
        if missing:
            metadata_missing += 1
            original_paths.append([])
        else:
            original_paths.append(path)

    leaf_support = collections.Counter()
    ancestor_support = collections.Counter()
    for item, path in enumerate(original_paths):
        if degrees[item] <= 0 or not path:
            continue
        leaf_support[tuple(path)] += 1
        for depth in range(1, len(path) + 1):
            ancestor_support[tuple(path[:depth])] += 1

    train_path = VERIFY_DIR / "train.txt"
    source_paths = {
        "train": relative(train_path),
        "mapping": relative(MAPPING_PATH),
        "merged_metadata": relative(MERGED_PATH),
    }
    source_hashes = {
        "train": sha256_file(train_path),
        "mapping": sha256_file(MAPPING_PATH),
        "merged_metadata": sha256_file(MERGED_PATH),
    }

    results = {}
    for threshold in THRESHOLDS:
        assignments = []
        provenance = []
        for item, path in enumerate(original_paths):
            original_name = " > ".join(path) if path else None
            final_name = None
            status = "metadata_missing"
            original_support = leaf_support[tuple(path)] if path else 0
            final_support = 0
            if path:
                if path == [ROOT_CATEGORY]:
                    status = "invalid_root_only"
                elif threshold is None or original_support >= threshold:
                    final_name = original_name
                    final_support = original_support
                    status = "retained"
                else:
                    chosen = choose_amazon_ancestor(path, ancestor_support, threshold)
                    if chosen is None:
                        status = "invalid_no_supported_ancestor"
                    else:
                        final_name = " > ".join(chosen)
                        final_support = ancestor_support[chosen]
                        status = "promoted"
            assignments.append(final_name)
            provenance.append(
                {
                    "item_id": item,
                    "train_degree": degrees[item],
                    "original_leaf": original_name,
                    "original_train_support": original_support,
                    "final_leaf": final_name,
                    "final_train_support": final_support,
                    "status": status,
                }
            )
        policy = {
            "kind": "amazon_hierarchical_train_support",
            "no_merge": threshold is None,
            "rare_definition": (
                None if threshold is None else "original leaf support < {}".format(threshold)
            ),
            "promotion": "nearest non-root ancestor with aggregated train support >= threshold",
            "root_only_arts_crafts_and_sewing_prototype": False,
            "invalid_policy": "retain in BPR/evaluation; exclude from prototype/EMA/taxonomy CL",
            "validation_or_test_used": False,
        }
        results[VARIANT_NAMES[threshold]] = finalize_taxonomy_variant(
            DATASET_NAME,
            VARIANT_NAMES[threshold],
            threshold,
            assignments,
            provenance,
            metadata_missing,
            sum(degree > 0 for degree in degrees),
            source_paths,
            source_hashes,
            policy,
        )[0]
    return results


def main():
    catalog_items = count_catalog_items()
    print("Catalog items (from item_list.txt): {}".format(catalog_items))

    validation_split = parse_split(VERIFY_DIR / "validation.txt", catalog_items)
    test_split = parse_split(VERIFY_DIR / "test.txt", catalog_items)
    train_split = parse_split(VERIFY_DIR / "train.txt", catalog_items)

    degrees = item_degrees(train_split["pairs"], catalog_items)
    masks = build_group_masks(degrees)

    sources = {
        "train": train_split,
        "validation": validation_split,
        "test": test_split,
    }
    _, protocol_manifest = build_evaluation_protocol(
        DATASET_NAME, degrees, masks, validation_split, test_split, sources
    )
    print("Evaluation protocol written for {}.".format(DATASET_NAME))
    print(json.dumps(protocol_manifest["output_hashes"], indent=2))

    variant_results = build_arts_crafts_and_sewing_taxonomy_variants(degrees, catalog_items)
    for variant_name, statistics in variant_results.items():
        print(
            "{}: catalog={} final_leaf_count={} coverage={}%".format(
                variant_name,
                statistics["catalog_items"],
                statistics["final_leaf_count"],
                statistics["coverage_percent"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
