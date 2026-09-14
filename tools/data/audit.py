"""Audit standalone data, evaluation and taxonomy artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config_path.config_path import evaluation_protocol_dir, verified_dataset_dir
from utility.utility_data.taxonomy import load_taxonomy, read_json, sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("amazon-book", "yelp2018"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--taxonomy-policies",
        nargs="+",
        default=["no_merge", "merge_t5", "merge_t10", "merge_t15"],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    protocol_directory = evaluation_protocol_dir(args.dataset, args.split_seed)
    manifest = read_json(protocol_directory / "manifest.json")
    checks = {}
    for split in ("train", "validation", "test"):
        actual = sha256_file(verified_dataset_dir(args.dataset) / (split + ".txt"))
        checks[split + "_hash"] = actual == manifest["source_hashes"][split]
    checks["group_mask_hash"] = (
        sha256_file(protocol_directory / "group_masks.npz")
        == manifest["output_hashes"]["group_masks.npz"]
    )
    variants = {}
    for policy in args.taxonomy_policies:
        taxonomy = load_taxonomy(args.dataset, policy)
        variants[policy] = {
            "taxonomy_hash": taxonomy.taxonomy_hash,
            "num_items": taxonomy.num_items,
            "num_prototypes": taxonomy.num_prototypes,
            "valid_items": int(taxonomy.valid_taxonomy_mask.sum()),
            "valid_train_items": int(taxonomy.valid_train_mask.sum()),
            "status": "PASS",
        }
    payload = {
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "checks": checks,
        "taxonomy_variants": variants,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    output = args.output or (
        ROOT / "results" / "data_audit" / (args.dataset + ".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**payload, "output": str(output)}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
