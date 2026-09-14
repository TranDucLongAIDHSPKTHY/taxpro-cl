"""Build deterministic 5-core train/validation/test splits from raw data."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import DATASET_DIR, DATASET_VERIFY_DIR, RESULT_DIR


DATASETS = ("amazon-book", "yelp2018")
SUPPORTED_SPLIT_SEEDS = (42, 123, 2026)
VALIDATION_RATIO = 0.1
ALGORITHM_ID = "taxprocl-iterative-kcore-per-user-shuffle-v2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all",) + DATASETS,
        default="all",
        help="Dataset to build (default: all).",
    )
    parser.add_argument("--minimum-degree", type=int, default=5)
    parser.add_argument(
        "--split-seed", type=int, choices=SUPPORTED_SPLIT_SEEDS, default=42
    )
    parser.add_argument("--source-root", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Staging root (default: results/data_splits/split_seed_<seed>).",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Write to dataset_verify; requires --force when files exist.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output split.",
    )
    return parser.parse_args(argv)


def read_interactions(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Interaction file not found: {}".format(path))
    pairs = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            tokens = line.split()
            if not tokens:
                continue
            try:
                values = [int(token) for token in tokens]
            except ValueError as error:
                raise ValueError(
                    "Non-integer token in {} at line {}".format(path, line_number)
                ) from error
            user, items = values[0], values[1:]
            if user < 0 or any(item < 0 for item in items):
                raise ValueError(
                    "Negative identifier in {} at line {}".format(path, line_number)
                )
            if len(items) != len(set(items)):
                raise ValueError(
                    "Duplicate item in {} at line {}".format(path, line_number)
                )
            for item in items:
                pair = (user, item)
                if pair in pairs:
                    raise ValueError(
                        "Duplicate interaction in {} at line {}".format(
                            path, line_number
                        )
                    )
                pairs.add(pair)
    if not pairs:
        raise ValueError("Interaction file is empty: {}".format(path))
    return pairs


def iterative_k_core(pairs, minimum_degree=5):
    if int(minimum_degree) <= 0:
        raise ValueError("minimum_degree must be positive")
    current = set(pairs)
    iterations = []
    while True:
        user_degrees = collections.Counter(user for user, _ in current)
        item_degrees = collections.Counter(item for _, item in current)
        filtered = {
            (user, item)
            for user, item in current
            if user_degrees[user] >= minimum_degree
            and item_degrees[item] >= minimum_degree
        }
        removed = len(current) - len(filtered)
        iterations.append(
            {
                "iteration": len(iterations) + 1,
                "input_interactions": len(current),
                "removed_interactions": removed,
                "output_interactions": len(filtered),
            }
        )
        current = filtered
        if removed == 0:
            break
    user_degrees = collections.Counter(user for user, _ in current)
    item_degrees = collections.Counter(item for _, item in current)
    statistics = {
        "minimum_degree": int(minimum_degree),
        "iterations": iterations,
        "interactions": len(current),
        "users": len(user_degrees),
        "items": len(item_degrees),
        "minimum_user_degree": min(user_degrees.values()) if user_degrees else 0,
        "minimum_item_degree": min(item_degrees.values()) if item_degrees else 0,
    }
    return current, statistics


def split_train_validation(
    pairs, validation_ratio=VALIDATION_RATIO, split_seed=42
):
    if float(validation_ratio) != VALIDATION_RATIO:
        raise ValueError("TaxPro-CL validation_ratio must be 0.1")
    if int(split_seed) not in SUPPORTED_SPLIT_SEEDS:
        raise ValueError("split_seed must be one of 42, 123, or 2026")
    by_user = collections.defaultdict(list)
    for user, item in pairs:
        by_user[user].append(item)
    train, validation = set(), set()
    random_generator = random.Random(int(split_seed))
    for user in sorted(by_user):
        items = sorted(by_user[user])
        random_generator.shuffle(items)
        validation_count = _validation_count(len(items), validation_ratio)
        held_out = set(items[:validation_count])
        validation.update((user, item) for item in held_out)
        train.update((user, item) for item in items if item not in held_out)
    if train & validation or train | validation != set(pairs):
        raise AssertionError("Train/validation partition contract failed")
    return train, validation


def build_dataset(
    dataset,
    source_root,
    output_root,
    minimum_degree=5,
    validation_ratio=VALIDATION_RATIO,
    split_seed=42,
    force=False,
    reference_root=None,
):
    source_directory = Path(source_root) / dataset
    output_directory = Path(output_root) / dataset
    train_source = source_directory / "train.txt"
    test_source = source_directory / "test.txt"
    outputs = {
        "train": output_directory / "train.txt",
        "validation": output_directory / "validation.txt",
        "test": output_directory / "test.txt",
        "manifest": output_directory / "split_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Output exists; use --force to replace it: {}".format(existing[0])
        )

    raw_train = read_interactions(train_source)
    raw_test = read_interactions(test_source)
    five_core, core_statistics = iterative_k_core(raw_train, minimum_degree)
    train, validation = split_train_validation(
        five_core,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
    )
    if five_core & raw_test:
        raise ValueError("Raw train and test contain overlapping interactions")

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_interactions_atomic(outputs["train"], train)
    _write_interactions_atomic(outputs["validation"], validation)
    _copy_atomic(test_source, outputs["test"])
    manifest = {
        "schema_version": 1,
        "algorithm": ALGORITHM_ID,
        "dataset": dataset,
        "split_seed": int(split_seed),
        "minimum_degree": int(minimum_degree),
        "validation_ratio": float(validation_ratio),
        "split_policy": {
            "user_order": "ascending_user_id",
            "item_order_before_shuffle": "ascending_item_id",
            "shuffle": "one_python_random_rng_per_dataset",
            "singleton_validation_count": 0,
            "degree_2_to_9_validation_count": 1,
            "degree_10_plus_validation_count": "floor(0.1 * N)",
            "supported_split_seeds": list(SUPPORTED_SPLIT_SEEDS),
        },
        "source_hashes": {
            "train": sha256_file(train_source),
            "test": sha256_file(test_source),
        },
        "output_hashes": {
            split: sha256_file(path)
            for split, path in outputs.items()
            if split != "manifest"
        },
        "counts": {
            "raw_train": len(raw_train),
            "five_core": len(five_core),
            "train": len(train),
            "validation": len(validation),
            "test": len(raw_test),
        },
        "five_core": core_statistics,
        "assertions": {
            "train_validation_disjoint": not bool(train & validation),
            "train_validation_equals_five_core": train | validation == five_core,
            "train_test_disjoint": not bool(train & raw_test),
            "validation_test_disjoint": not bool(validation & raw_test),
            "test_preserved_byte_for_byte": (
                sha256_file(test_source) == sha256_file(outputs["test"])
            ),
        },
    }
    if reference_root is not None:
        reference_directory = Path(reference_root) / dataset
        reference_hashes = {
            split: sha256_file(reference_directory / (split + ".txt"))
            for split in ("train", "validation", "test")
            if (reference_directory / (split + ".txt")).is_file()
        }
        matches = {
            split: manifest["output_hashes"].get(split) == digest
            for split, digest in reference_hashes.items()
        }
        manifest["locked_reference"] = {
            "hashes": reference_hashes,
            "matches": matches,
            "all_outputs_match": len(matches) == 3 and all(matches.values()),
        }
    if not all(manifest["assertions"].values()):
        raise AssertionError("Generated split failed its integrity assertions")
    _write_json_atomic(outputs["manifest"], manifest)
    return manifest


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if args.promote and args.output_root is not None:
        raise ValueError("--promote and --output-root are mutually exclusive")
    output_root = (
        DATASET_VERIFY_DIR
        if args.promote
        else args.output_root
        or RESULT_DIR / "data_splits" / ("split_seed_" + str(args.split_seed))
    )
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    manifests = [
        build_dataset(
            dataset,
            args.source_root,
            output_root,
            minimum_degree=args.minimum_degree,
            validation_ratio=VALIDATION_RATIO,
            split_seed=args.split_seed,
            force=args.force,
            reference_root=None if args.promote else DATASET_VERIFY_DIR,
        )
        for dataset in datasets
    ]
    payload = {
        "status": "PASS",
        "output_root": str(Path(output_root).resolve()),
        "promoted": bool(args.promote),
        "datasets": [
            {
                "dataset": manifest["dataset"],
                "counts": manifest["counts"],
                "output_hashes": manifest["output_hashes"],
            }
            for manifest in manifests
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validation_count(interaction_count, validation_ratio):
    if interaction_count <= 1:
        return 0
    if interaction_count < 10:
        return 1
    return int(interaction_count * validation_ratio)


def _write_interactions_atomic(path, pairs):
    by_user = collections.defaultdict(list)
    for user, item in pairs:
        by_user[user].append(item)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for user in sorted(by_user):
                items = " ".join(str(item) for item in sorted(by_user[user]))
                stream.write("{} {}\n".format(user, items))
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _copy_atomic(source, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copyfile(str(source), str(temporary))
        os.replace(str(temporary), str(destination))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
