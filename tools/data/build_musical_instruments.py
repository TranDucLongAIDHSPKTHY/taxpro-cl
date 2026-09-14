"""Build the musical-instruments dataset (third dataset candidate, after
video-games was rejected for having too thin a taxonomy -- median 1 item per
leaf category) from raw Amazon 5-core reviews.

Self-contained, additive-only companion to build_splits.py: this file exists
specifically so amazon-book/yelp2018's pipeline files never need to be
touched. It imports and reuses build_splits.py's pure helper functions
(iterative_k_core, sha256_file, the atomic-write helpers) instead of
duplicating them, but owns its own dataset name, paths, and split ratio --
none of which build_splits.py's DATASETS tuple / hardcoded 0.1
validation_ratio could express without editing that shared file.

Pipeline (same policy as build_video_games.py before it): download the
5-core Musical_Instruments review file directly from McAuley Lab (5-core
filtering already applied upstream), verify 5-core holds after our own ID
remap, then do ONE deterministic per-user 70/10/20 shuffle-split (single
seeded RNG reused across users, sorted-user iteration order -- same "one RNG
per dataset" policy build_splits.py documents for its own validation carve).
No second k-core re-filter pass after the split.

Outputs, mirroring the amazon-book/yelp2018/video-games two-stage convention:
    dataset/musical-instruments/{train,test,item_list,user_list}.txt
    dataset_verify/musical-instruments/{train,validation,test}.txt + split_manifest.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data.build_splits import (  # noqa: E402  (reused, not duplicated)
    iterative_k_core,
    sha256_file,
    _write_interactions_atomic,
    _write_json_atomic,
)

DATASET_NAME = "musical-instruments"
ALGORITHM_ID = "taxprocl-musical-instruments-per-user-shuffle-70-10-20-v1"
SPLIT_SEED = 42
RATIOS = {"train": 0.7, "validation": 0.1, "test": 0.2}
MINIMUM_DEGREE = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews-file",
        type=Path,
        required=True,
        help="Path to the decompressed 5-core Musical_Instruments review JSON-lines file.",
    )
    parser.add_argument("--raw-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--verify-root", type=Path, default=ROOT / "dataset_verify")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--minimum-degree", type=int, default=MINIMUM_DEGREE)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output files."
    )
    return parser.parse_args(argv)


def read_reviews(path):
    """Yield (reviewer_id, asin) for every parseable review line."""
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            reviewer = record.get("reviewerID")
            asin = record.get("asin")
            if not reviewer or not asin:
                raise ValueError(
                    "Missing reviewerID/asin at {}:{}".format(path, line_number)
                )
            yield reviewer, asin


def build_id_maps(pairs):
    """Deterministic org_id -> remap_id, sorted lexicographically (matches
    the LightGCN convention's determinism, independent of file line order)."""
    users = sorted({user for user, _ in pairs})
    items = sorted({item for _, item in pairs})
    user_map = {org_id: index for index, org_id in enumerate(users)}
    item_map = {org_id: index for index, org_id in enumerate(items)}
    return user_map, item_map


def write_id_list(path, id_map):
    ordered = sorted(id_map.items(), key=lambda pair: pair[1])
    lines = ["org_id remap_id"] + [
        "{} {}".format(org_id, remap_id) for org_id, remap_id in ordered
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def per_user_split(pairs, split_seed, ratios):
    """One seeded RNG, sorted-user iteration -- same determinism policy as
    build_splits.py's split_train_validation. round() (banker's rounding)
    on train/validation counts; test takes the exact remainder so the
    three counts always sum to the user's true degree, no drift."""
    by_user = collections.defaultdict(list)
    for user, item in pairs:
        by_user[user].append(item)
    train, validation, test = set(), set(), set()
    rng = random.Random(int(split_seed))
    for user in sorted(by_user):
        items = sorted(by_user[user])
        rng.shuffle(items)
        degree = len(items)
        train_count = round(ratios["train"] * degree)
        validation_count = round(ratios["validation"] * degree)
        train_count = min(train_count, degree)
        validation_count = min(validation_count, degree - train_count)
        test_count = degree - train_count - validation_count
        cursor = 0
        train.update((user, item) for item in items[cursor:cursor + train_count])
        cursor += train_count
        validation.update(
            (user, item) for item in items[cursor:cursor + validation_count]
        )
        cursor += validation_count
        test.update((user, item) for item in items[cursor:cursor + test_count])
    return train, validation, test


def main(argv=None):
    args = parse_args(argv)

    raw_directory = args.raw_root / DATASET_NAME
    verify_directory = args.verify_root / DATASET_NAME
    outputs = {
        "raw_train": raw_directory / "train.txt",
        "raw_test": raw_directory / "test.txt",
        "item_list": raw_directory / "item_list.txt",
        "user_list": raw_directory / "user_list.txt",
        "train": verify_directory / "train.txt",
        "validation": verify_directory / "validation.txt",
        "test": verify_directory / "test.txt",
        "manifest": verify_directory / "split_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Output exists; use --force to replace it: {}".format(existing[0])
        )

    print("Reading reviews from {} ...".format(args.reviews_file))
    org_pairs = set(read_reviews(args.reviews_file))
    print("Distinct (reviewer, asin) pairs: {}".format(len(org_pairs)))

    user_map, item_map = build_id_maps(org_pairs)
    print(
        "Distinct users: {}  Distinct items: {}".format(
            len(user_map), len(item_map)
        )
    )

    remapped_pairs = {
        (user_map[user], item_map[item]) for user, item in org_pairs
    }

    five_core, core_statistics = iterative_k_core(
        remapped_pairs, args.minimum_degree
    )
    print("5-core verification: {}".format(core_statistics))
    dropped = len(remapped_pairs) - len(five_core)
    if dropped:
        print(
            "WARNING: {} interactions dropped re-verifying 5-core after ID remap "
            "(source file claims pre-filtered 5-core; this is a safety re-check, "
            "not expected to remove much).".format(dropped)
        )

    train, validation, test = per_user_split(five_core, args.split_seed, RATIOS)
    if train & validation or train & test or validation & test:
        raise AssertionError("Split partitions are not disjoint")
    if train | validation | test != five_core:
        raise AssertionError("Split partition does not cover the 5-core pool")

    raw_directory.mkdir(parents=True, exist_ok=True)
    verify_directory.mkdir(parents=True, exist_ok=True)

    write_id_list(outputs["user_list"], user_map)
    write_id_list(outputs["item_list"], item_map)
    _write_interactions_atomic(outputs["raw_train"], train | validation)
    _write_interactions_atomic(outputs["raw_test"], test)
    _write_interactions_atomic(outputs["train"], train)
    _write_interactions_atomic(outputs["validation"], validation)
    _write_interactions_atomic(outputs["test"], test)

    manifest = {
        "schema_version": 1,
        "algorithm": ALGORITHM_ID,
        "dataset": DATASET_NAME,
        "split_seed": int(args.split_seed),
        "minimum_degree": int(args.minimum_degree),
        "ratios": RATIOS,
        "split_policy": {
            "user_order": "ascending_user_id",
            "item_order_before_shuffle": "ascending_item_id",
            "shuffle": "one_python_random_rng_per_dataset",
            "count_rounding": "round(ratio * degree) for train/validation, "
            "test takes the exact remainder",
        },
        "source": {
            "reviews_file": str(args.reviews_file),
            "reviews_file_sha256": sha256_file(args.reviews_file),
        },
        "id_maps": {
            "user_list_sha256": sha256_file(outputs["user_list"]),
            "item_list_sha256": sha256_file(outputs["item_list"]),
        },
        "counts": {
            "distinct_review_pairs": len(org_pairs),
            "five_core_pairs": len(five_core),
            "dropped_reverifying_five_core": dropped,
            "users": len(user_map),
            "items": len(item_map),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "five_core_verification": core_statistics,
        "output_hashes": {
            name: sha256_file(outputs[name])
            for name in ("raw_train", "raw_test", "train", "validation", "test")
        },
        "assertions": {
            "train_validation_disjoint": not bool(train & validation),
            "train_test_disjoint": not bool(train & test),
            "validation_test_disjoint": not bool(validation & test),
            "partition_covers_five_core": train | validation | test == five_core,
        },
    }
    if not all(manifest["assertions"].values()):
        raise AssertionError("Generated split failed its integrity assertions")
    _write_json_atomic(outputs["manifest"], manifest)

    print("Wrote raw stage to {}".format(raw_directory))
    print("Wrote processed stage to {}".format(verify_directory))
    print(
        "Final split: train={} validation={} test={} (target ratios {})".format(
            len(train), len(validation), len(test), RATIOS
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
