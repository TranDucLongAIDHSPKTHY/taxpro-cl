"""Build and audit all TaxPro-CL protocol artifacts.

The script uses only the Python standard library.  It never writes dataset/
or dataset_verify/ and never uses validation/test statistics to define item
groups or taxonomy variants.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data.build_splits import iterative_k_core


SPLIT_SEED = 42
PROTOCOL_VERSION = "taxprocl-p0-week3-v1"
THRESHOLDS = (None, 5, 10, 15)
VARIANT_NAMES = {
    None: "no_merge",
    5: "merge_t5",
    10: "merge_t10",
    15: "merge_t15",
}
DATASETS = {
    "amazon-book": {
        "catalog_items": 91599,
        "raw": ROOT / "dataset" / "amazon-book",
        "verify": ROOT / "dataset_verify" / "amazon-book",
        "mapping": ROOT / "metadata" / "item2category_amazon.json",
        "merged": ROOT / "metadata" / "merged_metadata_amazon.json",
        "taxonomy_kind": "hierarchical",
    },
    "yelp2018": {
        "catalog_items": 38048,
        "raw": ROOT / "dataset" / "yelp2018",
        "verify": ROOT / "dataset_verify" / "yelp2018",
        "mapping": ROOT / "metadata" / "item2category_yelp.json",
        "merged": ROOT / "metadata" / "merged_metadata_yelp.json",
        "taxonomy_kind": "flat_multilabel",
    },
}
GROUP_DEFINITIONS = {
    "near_cold": {"minimum": 1, "maximum": 5},
    "long_tail": {"minimum": 1, "maximum": 10},
    "warm": {"minimum": 11, "maximum": None},
    "strict_cold": {"minimum": 0, "maximum": 0, "audit_only": True},
}
METRIC_NAMES = [
    "{}@{}_{}".format(metric, k, group)
    for group in ("Overall", "NearCold", "LongTail", "Warm")
    for metric in ("Precision", "Recall", "NDCG")
    for k in (10, 20)
]


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")
    )


def write_compact_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")



def parse_split(path, catalog_items):
    by_user = {}
    pair_set = set()
    duplicate_pairs = 0
    duplicate_items_within_line = 0
    duplicate_user_lines = 0
    empty_user_lines = 0
    blank_lines = 0
    malformed = []
    max_user = -1
    max_item = -1
    user_line_counts = collections.Counter()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            stripped = raw_line.strip()
            if not stripped:
                blank_lines += 1
                continue
            try:
                values = [int(token) for token in stripped.split()]
            except ValueError:
                malformed.append({"line": line_number, "reason": "non_integer"})
                continue
            user = values[0]
            items = values[1:]
            if user < 0:
                malformed.append({"line": line_number, "reason": "negative_user"})
                continue
            user_line_counts[user] += 1
            if user_line_counts[user] > 1:
                duplicate_user_lines += 1
            max_user = max(max_user, user)
            if not items:
                empty_user_lines += 1
                by_user.setdefault(user, [])
                continue
            seen_line = set()
            target = by_user.setdefault(user, [])
            for item in items:
                if item < 0 or item >= catalog_items:
                    malformed.append(
                        {
                            "line": line_number,
                            "reason": "item_out_of_range",
                            "item": item,
                        }
                    )
                    continue
                max_item = max(max_item, item)
                if item in seen_line:
                    duplicate_items_within_line += 1
                seen_line.add(item)
                pair = (user, item)
                if pair in pair_set:
                    duplicate_pairs += 1
                else:
                    pair_set.add(pair)
                    target.append(item)
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "by_user": by_user,
        "pairs": pair_set,
        "users_with_records": len(by_user),
        "users_with_interactions": sum(bool(items) for items in by_user.values()),
        "items_with_interactions": len({item for _, item in pair_set}),
        "interactions": len(pair_set),
        "max_user_id": max_user,
        "max_item_id": max_item,
        "blank_lines": blank_lines,
        "empty_user_lines": empty_user_lines,
        "duplicate_user_lines": duplicate_user_lines,
        "duplicate_items_within_line": duplicate_items_within_line,
        "duplicate_interactions": duplicate_pairs,
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:20],
    }


def serializable_split(split):
    return {
        key: value
        for key, value in split.items()
        if key not in {"by_user", "pairs"}
    }


def item_degrees(train_pairs, catalog_items):
    degrees = [0] * catalog_items
    for _, item in train_pairs:
        degrees[item] += 1
    return degrees


def build_group_masks(degrees):
    masks = {
        "near_cold": [1 <= degree <= 5 for degree in degrees],
        "long_tail": [1 <= degree <= 10 for degree in degrees],
        "warm": [degree > 10 for degree in degrees],
        "strict_cold": [degree == 0 for degree in degrees],
    }
    if not all(
        (not near) or long
        for near, long in zip(masks["near_cold"], masks["long_tail"])
    ):
        raise AssertionError("near_cold must be a subset of long_tail")
    if any(
        warm and long
        for warm, long in zip(masks["warm"], masks["long_tail"])
    ):
        raise AssertionError("warm must be disjoint from long_tail")
    if any(
        cold and (near or long or warm)
        for cold, near, long, warm in zip(
            masks["strict_cold"],
            masks["near_cold"],
            masks["long_tail"],
            masks["warm"],
        )
    ):
        raise AssertionError("strict_cold must be outside the three evaluation slices")
    return masks


def targets_by_group(split_by_user, masks):
    targets = {
        "overall": {},
        "near_cold": {},
        "long_tail": {},
        "warm": {},
        "strict_cold": {},
    }
    for user in sorted(split_by_user):
        items = split_by_user[user]
        if items:
            targets["overall"][str(user)] = list(items)
        for group in ("near_cold", "long_tail", "warm", "strict_cold"):
            selected = [item for item in items if masks[group][item]]
            if selected:
                targets[group][str(user)] = selected
    return targets


def target_support(targets):
    return {
        "eligible_users": len(targets),
        "relevant_items": len(
            {item for items in targets.values() for item in items}
        ),
        "positive_interactions": sum(len(items) for items in targets.values()),
    }


def npy_bytes(values, descriptor):
    if descriptor == "|b1":
        payload = bytes(1 if value else 0 for value in values)
    elif descriptor == "<i8":
        payload = struct.pack("<{}q".format(len(values)), *values)
    else:
        raise ValueError("Unsupported NPY descriptor")
    header = "{'descr': '%s', 'fortran_order': False, 'shape': (%d,), }" % (
        descriptor,
        len(values),
    )
    header_bytes = header.encode("latin-1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    return b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(header_bytes)) + header_bytes + payload


def write_npy(path, values, descriptor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(npy_bytes(values, descriptor))


def write_masks_npz(path, masks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for group in ("near_cold", "long_tail", "warm", "strict_cold"):
            info = zipfile.ZipInfo(group + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(masks[group], "|b1"))


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": 0, "median": 0, "max": 0, "mean": 0.0}
    middle = len(values) // 2
    median = (
        values[middle]
        if len(values) % 2
        else (values[middle - 1] + values[middle]) / 2.0
    )
    return {
        "count": len(values),
        "min": values[0],
        "median": median,
        "max": values[-1],
        "mean": round(sum(values) / len(values), 6),
    }


def group_statistics(degrees, masks, validation_targets, test_targets):
    catalog = len(degrees)
    observed = sum(degree > 0 for degree in degrees)
    result = {
        "catalog_items": catalog,
        "train_observed_items": observed,
        "groups": {},
        "assertions": {
            "near_cold_subset_long_tail": all(
                (not near) or long
                for near, long in zip(masks["near_cold"], masks["long_tail"])
            ),
            "warm_disjoint_long_tail": not any(
                warm and long
                for warm, long in zip(masks["warm"], masks["long_tail"])
            ),
            "strict_cold_outside_p0_groups": not any(
                cold and (near or long or warm)
                for cold, near, long, warm in zip(
                    masks["strict_cold"],
                    masks["near_cold"],
                    masks["long_tail"],
                    masks["warm"],
                )
            ),
        },
    }
    for group in ("near_cold", "long_tail", "warm", "strict_cold"):
        count = sum(masks[group])
        result["groups"][group] = {
            "definition": GROUP_DEFINITIONS[group],
            "item_count": count,
            "percent_of_catalog": round(100.0 * count / catalog, 6),
            "percent_of_train_observed": (
                round(100.0 * count / observed, 6) if observed else 0.0
            ),
            "validation": target_support(validation_targets[group]),
            "test": target_support(test_targets[group]),
        }
    return result


def output_hashes(directory, names):
    return {
        name: sha256_file(Path(directory) / name)
        for name in sorted(names)
    }


def build_evaluation_protocol(
    dataset_name, degrees, masks, validation, test, sources, output_root=None
):
    output = (
        Path(output_root) / dataset_name
        if output_root is not None
        else ROOT / "preprocessed" / "evaluation_protocol" / "split_seed_42" / dataset_name
    )
    output.mkdir(parents=True, exist_ok=True)
    degree_payload = {
        "dataset": dataset_name,
        "split_seed": SPLIT_SEED,
        "definition": "number of unique (user,item) interactions in model train only",
        "degrees": degrees,
    }
    validation_targets = targets_by_group(validation["by_user"], masks)
    test_targets = targets_by_group(test["by_user"], masks)
    statistics = group_statistics(
        degrees, masks, validation_targets, test_targets
    )
    write_compact_json(output / "item_degrees_train.json", degree_payload)
    write_masks_npz(output / "group_masks.npz", masks)
    write_json(output / "validation_targets_by_group.json", validation_targets)
    write_json(output / "test_targets_by_group.json", test_targets)
    write_json(output / "group_statistics.json", statistics)
    names = [
        "item_degrees_train.json",
        "group_masks.npz",
        "validation_targets_by_group.json",
        "test_targets_by_group.json",
        "group_statistics.json",
    ]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset_name,
        "split_seed": SPLIT_SEED,
        "group_definitions": GROUP_DEFINITIONS,
        "ranking_protocol": {
            "candidate_catalog": "full catalog",
            "group_slice": "filter relevant positives only",
            "denominator": "users with >=1 relevant positive in the slice",
            "validation_excludes": "train positives",
            "test_excludes": "train + validation positives",
            "test_role": "evaluation-only",
        },
        "source_paths": {key: value["path"] for key, value in sources.items()},
        "source_hashes": {key: value["sha256"] for key, value in sources.items()},
        "creation_command": (
            "python -m tools.protocol.build --build"
        ),
        "output_hashes": output_hashes(output, names),
    }
    write_json(output / "manifest.json", manifest)
    return statistics, manifest


def sanitize_path(value):
    if not isinstance(value, list):
        return []
    return [str(part).strip() for part in value if str(part).strip()]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def taxonomy_source(dataset_name):
    config = DATASETS[dataset_name]
    mapping = load_json(config["mapping"])
    merged = load_json(config["merged"])
    return mapping, merged


def taxonomy_record(item, mapping):
    record = mapping.get(str(item))
    if not record:
        return None, []
    if isinstance(record, dict):
        original = sanitize_path(
            record.get("original_taxonomy_path") or record.get("taxonomy_path")
        )
        return record, original
    if isinstance(record, list):
        for candidate in record:
            path = sanitize_path(candidate)
            if path:
                return None, path
    return None, []


def choose_amazon_ancestor(path, ancestor_support, threshold):
    """Choose the nearest supported non-root ancestor; never return root Books."""
    for depth in range(len(path) - 1, 1, -1):
        candidate = tuple(path[:depth])
        if ancestor_support.get(candidate, 0) >= threshold:
            return candidate
    return None


def choose_yelp_candidate(categories, primary, support, threshold):
    """Choose the most specific supported same-business alternative deterministically."""
    candidates = [
        category
        for category in categories
        if category != primary and support.get(category, 0) >= threshold
    ]
    candidates.sort(key=lambda category: (support.get(category, 0), category))
    return candidates[0] if candidates else None


def finalize_taxonomy_variant(
    dataset_name,
    variant_name,
    threshold,
    assignments,
    provenance,
    metadata_missing,
    train_observed_count,
    source_paths,
    source_hashes,
    policy,
):
    output = ROOT / "metadata" / "taxonomy_variants" / dataset_name / variant_name
    output.mkdir(parents=True, exist_ok=True)
    valid_names = sorted({name for name in assignments if name is not None})
    leaf_id = {name: index for index, name in enumerate(valid_names)}
    ids = [leaf_id[name] if name is not None else -1 for name in assignments]
    mask = [name is not None for name in assignments]
    item_payload = {
        str(item): {
            "leaf_id": ids[item],
            "leaf_name": assignments[item],
            "valid_taxonomy": mask[item],
        }
        for item in range(len(assignments))
    }
    write_compact_json(output / "item_to_leaf.json", item_payload)
    write_npy(output / "item_to_leaf_id.npy", ids, "<i8")
    write_npy(output / "valid_taxonomy_mask.npy", mask, "|b1")
    write_json(
        output / "leaf_id_to_name.json",
        {str(index): name for index, name in enumerate(valid_names)},
    )
    with (output / "mapping_provenance.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in provenance:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    final_support = collections.Counter(
        assignments[item]
        for item in range(len(assignments))
        if assignments[item] is not None and provenance[item]["train_degree"] > 0
    )
    statuses = collections.Counter(row["status"] for row in provenance)
    statistics = {
        "dataset": dataset_name,
        "variant": variant_name,
        "threshold": threshold,
        "policy": policy,
        "catalog_items": len(assignments),
        "train_observed_items": train_observed_count,
        "original_leaf_count": len(
            {
                row["original_leaf"]
                for row in provenance
                if row["original_leaf"] is not None
            }
        ),
        "final_leaf_count": len(valid_names),
        "retained_items": statuses["retained"],
        "promoted_items": statuses["promoted"],
        "reassigned_items": statuses["reassigned"],
        "invalid_items": sum(not value for value in mask),
        "metadata_missing_items": metadata_missing,
        "coverage_percent": round(100.0 * sum(mask) / len(mask), 6),
        "final_leaf_train_support_distribution": distribution(
            list(final_support.values())
        ),
        "status_counts": dict(sorted(statuses.items())),
    }
    write_json(output / "taxonomy_statistics.json", statistics)
    output_names = [
        "item_to_leaf.json",
        "item_to_leaf_id.npy",
        "valid_taxonomy_mask.npy",
        "leaf_id_to_name.json",
        "mapping_provenance.jsonl",
        "taxonomy_statistics.json",
    ]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset_name,
        "variant": variant_name,
        "threshold": threshold,
        "policy": policy,
        "source_paths": source_paths,
        "source_hashes": source_hashes,
        "creation_command": "python -m tools.protocol.build --build",
        "output_hashes": output_hashes(output, output_names),
    }
    write_json(output / "manifest.json", manifest)
    return statistics, manifest


def build_amazon_variants(degrees):
    dataset_name = "amazon-book"
    config = DATASETS[dataset_name]
    mapping, merged = taxonomy_source(dataset_name)
    original_paths = []
    metadata_missing = 0
    for item in range(config["catalog_items"]):
        record, path = taxonomy_record(item, mapping)
        merged_record = merged.get(str(item))
        merged_paths = (
            [
                sanitize_path(candidate)
                for candidate in (merged_record or {}).get("taxonomy_paths") or []
                if sanitize_path(candidate)
            ]
        )
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
    results = {}
    source_paths = {
        "train": relative(config["verify"] / "train.txt"),
        "mapping": relative(config["mapping"]),
        "merged_metadata": relative(config["merged"]),
    }
    source_hashes = {
        "train": sha256_file(config["verify"] / "train.txt"),
        "mapping": sha256_file(config["mapping"]),
        "merged_metadata": sha256_file(config["merged"]),
    }
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
                if path == ["Books"]:
                    status = "invalid_root_only"
                elif threshold is None or original_support >= threshold:
                    final_name = original_name
                    final_support = original_support
                    status = "retained"
                else:
                    chosen = choose_amazon_ancestor(
                        path, ancestor_support, threshold
                    )
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
            "root_only_books_prototype": False,
            "invalid_policy": "retain in BPR/evaluation; exclude from prototype/EMA/taxonomy CL",
            "validation_or_test_used": False,
        }
        results[VARIANT_NAMES[threshold]] = finalize_taxonomy_variant(
            dataset_name,
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


def yelp_categories(record):
    categories = []
    if record:
        for path in record.get("taxonomy_paths") or []:
            clean = sanitize_path(path)
            if clean:
                categories.append(clean[-1])
    return sorted(set(categories))


def build_yelp_variants(degrees):
    dataset_name = "yelp2018"
    config = DATASETS[dataset_name]
    mapping, merged = taxonomy_source(dataset_name)
    primary = []
    all_categories = []
    metadata_missing_flags = []
    for item in range(config["catalog_items"]):
        record, path = taxonomy_record(item, mapping)
        merged_record = merged.get(str(item))
        categories = yelp_categories(merged_record)
        current_primary = path[-1] if path else (categories[0] if categories else None)
        missing = (
            not merged_record
            or merged_record.get("status") == "missing"
            or not categories
        )
        metadata_missing_flags.append(missing)
        primary.append(None if missing else current_primary)
        all_categories.append([] if missing else categories)
    support = collections.Counter()
    for item, categories in enumerate(all_categories):
        if degrees[item] > 0:
            support.update(categories)
    results = {}
    source_paths = {
        "train": relative(config["verify"] / "train.txt"),
        "mapping": relative(config["mapping"]),
        "merged_metadata": relative(config["merged"]),
    }
    source_hashes = {
        "train": sha256_file(config["verify"] / "train.txt"),
        "mapping": sha256_file(config["mapping"]),
        "merged_metadata": sha256_file(config["merged"]),
    }
    for threshold in THRESHOLDS:
        assignments = []
        provenance = []
        for item, current_primary in enumerate(primary):
            original_support = support[current_primary] if current_primary else 0
            final_name = None
            final_support = 0
            status = "metadata_missing" if metadata_missing_flags[item] else "invalid_missing_primary"
            if current_primary is not None:
                if threshold is None or original_support >= threshold:
                    final_name = current_primary
                    final_support = original_support
                    status = "retained"
                else:
                    candidate = choose_yelp_candidate(
                        all_categories[item],
                        current_primary,
                        support,
                        threshold,
                    )
                    if candidate:
                        final_name = candidate
                        final_support = support[final_name]
                        status = "reassigned"
                    else:
                        status = "invalid_no_supported_business_category"
            assignments.append(final_name)
            provenance.append(
                {
                    "item_id": item,
                    "train_degree": degrees[item],
                    "business_categories": all_categories[item],
                    "original_leaf": current_primary,
                    "original_train_support": original_support,
                    "final_leaf": final_name,
                    "final_train_support": final_support,
                    "status": status,
                }
            )
        policy = {
            "kind": "yelp_flat_multilabel_train_support",
            "no_merge": threshold is None,
            "rare_definition": (
                None if threshold is None else "primary category support < {}".format(threshold)
            ),
            "reassignment": (
                "same-business category with smallest train support >= threshold; "
                "category-name tie-break"
            ),
            "shared_other_or_unknown_prototype": False,
            "invalid_policy": "retain in BPR/evaluation; exclude from prototype/EMA/taxonomy CL",
            "validation_or_test_used": False,
        }
        results[VARIANT_NAMES[threshold]] = finalize_taxonomy_variant(
            dataset_name,
            VARIANT_NAMES[threshold],
            threshold,
            assignments,
            provenance,
            sum(metadata_missing_flags),
            sum(degree > 0 for degree in degrees),
            source_paths,
            source_hashes,
            policy,
        )[0]
    return results


def build_experiment_protocol(audit, taxonomy_tables):
    variants = [
        {"dataset": dataset, "policy": variant}
        for dataset in ("amazon-book", "yelp2018")
        for variant in ("no_merge", "merge_t5", "merge_t10", "merge_t15")
    ]
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "Protocol/split preparation only; no full baseline or TaxPro-CL training",
        "data_split_seed": SPLIT_SEED,
        "future_training_seeds": [42, 0, 1],
        "metrics": METRIC_NAMES,
        "group_definitions": GROUP_DEFINITIONS,
        "evaluation": {
            "overall_is_default": True,
            "full_candidate_catalog": True,
            "group_ground_truth_filter_only": True,
            "eligible_user_denominator": "users with at least one target positive in group",
            "validation_exclusions": ["train positives"],
            "test_exclusions": ["train positives", "validation positives"],
        },
        "taxonomy_configurations": variants,
        "threshold_selection": {
            "test_used": False,
            "control": "no_merge",
            "primary_metric": "validation Recall@20 Overall",
            "best_checkpoint_metric": "validation Recall@20 Overall",
            "group_metrics_role": "analysis and guardrail only",
            "near_tie_tolerance_absolute_recall_at_20": 0.0005,
            "least_intervention_order": [
                "no_merge",
                "merge_t5",
                "merge_t10",
                "merge_t15",
            ],
        },
        "future_run_plan": {
            "week_5": (
                "TaxProCL smoke/integration training with no_merge and one threshold "
                "variant after stability infrastructure is ready"
            ),
            "week_6_screening": "4 policies x 2 datasets x seed 42 = 8 runs",
            "full_if_budget": "4 policies x 2 datasets x seeds {42,0,1} = 24 runs",
            "limited_compute": (
                "run 8 screening runs; retain no_merge and best validation policy; "
                "add seeds 0 and 1; label screening vs final"
            ),
        },
        "dataset_hashes": {
            dataset["dataset"]: {
                split: dataset["splits"][split]["sha256"]
                for split in ("train", "validation", "test")
            }
            for dataset in audit["datasets"]
        },
        "taxonomy_summary": taxonomy_tables,
    }
    write_json(ROOT / "experiment_protocol_v1.json", protocol)
    return protocol




def build_audit():
    test_before = {
        dataset: {
            "raw": sha256_file(config["raw"] / "test.txt"),
            "verify": sha256_file(config["verify"] / "test.txt"),
        }
        for dataset, config in DATASETS.items()
    }
    audit_datasets = []
    runtime = {}
    failures = []
    for dataset_name, config in DATASETS.items():
        print("Auditing {}...".format(dataset_name), flush=True)
        splits = {
            split: parse_split(
                config["verify"] / (split + ".txt"),
                config["catalog_items"],
            )
            for split in ("train", "validation", "test")
        }
        raw_train = parse_split(
            config["raw"] / "train.txt", config["catalog_items"]
        )
        reconstructed, core_stats = iterative_k_core(raw_train["pairs"])
        model_union = splits["train"]["pairs"] | splits["validation"]["pairs"]
        overlaps = {
            "train_validation": len(
                splits["train"]["pairs"] & splits["validation"]["pairs"]
            ),
            "train_test": len(splits["train"]["pairs"] & splits["test"]["pairs"]),
            "validation_test": len(
                splits["validation"]["pairs"] & splits["test"]["pairs"]
            ),
        }
        degrees = item_degrees(
            splits["train"]["pairs"], config["catalog_items"]
        )
        masks = build_group_masks(degrees)
        group_stats, evaluation_manifest = build_evaluation_protocol(
            dataset_name,
            degrees,
            masks,
            splits["validation"],
            splits["test"],
            splits,
        )
        mapping = load_json(config["mapping"])
        valid_mapping_ids = [
            int(item)
            for item in mapping
            if str(item).isdigit() and 0 <= int(item) < config["catalog_items"]
        ]
        valid_taxonomy_ids = [
            item
            for item in valid_mapping_ids
            if taxonomy_record(item, mapping)[1]
        ]
        merged = load_json(config["merged"])
        merged_taxonomy_items = sum(
            1
            for item in range(config["catalog_items"])
            if merged.get(str(item))
            and merged[str(item)].get("status") != "missing"
            and any(
                sanitize_path(path)
                for path in merged[str(item)].get("taxonomy_paths") or []
            )
        )
        metadata_source_missing = sum(
            1
            for item in range(config["catalog_items"])
            if not merged.get(str(item))
            or merged[str(item)].get("status") == "missing"
            or not merged[str(item)].get("taxonomy_paths")
        )
        taxonomy_coverage = {
            "source_path": relative(config["mapping"]),
            "source_sha256": sha256_file(config["mapping"]),
            "mapping_records": len(mapping),
            "valid_item_ids": len(valid_mapping_ids),
            "items_with_valid_taxonomy": len(valid_taxonomy_ids),
            "invalid_or_out_of_range_ids": len(mapping) - len(valid_mapping_ids),
            "catalog_items": config["catalog_items"],
            "coverage_percent": round(
                100.0 * len(valid_taxonomy_ids) / config["catalog_items"], 6
            ),
            "merged_metadata_items_with_taxonomy": merged_taxonomy_items,
            "merged_metadata_taxonomy_coverage_percent": round(
                100.0 * merged_taxonomy_items / config["catalog_items"], 6
            ),
            "missing_or_empty_taxonomy_items": (
                config["catalog_items"] - len(valid_taxonomy_ids)
            ),
            "metadata_source_missing_items": metadata_source_missing,
        }
        assertions = {
            "split_format_valid": all(
                split["malformed_count"] == 0
                and split["duplicate_interactions"] == 0
                for split in splits.values()
            ),
            "zero_split_leakage": all(value == 0 for value in overlaps.values()),
            "raw_test_equals_verify_test": (
                test_before[dataset_name]["raw"]
                == test_before[dataset_name]["verify"]
            ),
            "train_plus_validation_equals_reconstructed_five_core": (
                model_union == reconstructed
            ),
            "model_train_validation_disjoint": not (
                splits["train"]["pairs"] & splits["validation"]["pairs"]
            ),
            "mapping_ids_valid": (
                len(valid_mapping_ids) == len(mapping)
            ),
            **group_stats["assertions"],
        }
        if not all(assertions.values()):
            failures.extend(
                "{}:{}".format(dataset_name, key)
                for key, value in assertions.items()
                if not value
            )
        audit_datasets.append(
            {
                "dataset": dataset_name,
                "catalog_items": config["catalog_items"],
                "splits": {
                    split: serializable_split(payload)
                    for split, payload in splits.items()
                },
                "raw_train": serializable_split(raw_train),
                "five_core": core_stats,
                "split_overlap_interactions": overlaps,
                "taxonomy_coverage": taxonomy_coverage,
                "groups": group_stats,
                "evaluation_manifest_sha256": sha256_file(
                    ROOT
                    / "preprocessed"
                    / "evaluation_protocol"
                    / "split_seed_42"
                    / dataset_name
                    / "manifest.json"
                ),
                "assertions": assertions,
            }
        )
        runtime[dataset_name] = {
            "degrees": degrees,
            "masks": masks,
            "splits": splits,
            "evaluation_manifest": evaluation_manifest,
        }
    test_after = {
        dataset: {
            "raw": sha256_file(config["raw"] / "test.txt"),
            "verify": sha256_file(config["verify"] / "test.txt"),
        }
        for dataset, config in DATASETS.items()
    }
    for dataset in DATASETS:
        if test_before[dataset] != test_after[dataset]:
            failures.append(dataset + ":test_hash_changed_during_build")
    audit = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "repository_root": ".",  # relative by design; do not bake in an absolute local path
        "protocol": {
            "data_split_seed": SPLIT_SEED,
            "future_training_seeds": [42, 0, 1],
            "groups": GROUP_DEFINITIONS,
            "group_source": "model train only",
            "test_role": "evaluation-only",
        },
        "datasets": audit_datasets,
        "test_hashes_before": test_before,
        "test_hashes_after": test_after,
        "gate_g1": {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
        },
    }
    return audit, runtime


def build_audit_markdown(audit):
    lines = [
        "# DATA PROTOCOL AUDIT - TAXPRO-CL",
        "",
        "**Protocol:** `{}`  ".format(audit["protocol_version"]),
        "**Data split seed:** 42  ",
        "**Gate G1:** **{}**".format(audit["gate_g1"]["status"]),
        "",
        "## Splits, hashes, and leakage",
        "",
        "| Dataset | Split | Users | Items | Interactions | SHA-256 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for dataset in audit["datasets"]:
        for split in ("train", "validation", "test"):
            row = dataset["splits"][split]
            lines.append(
                "| {} | {} | {:,} | {:,} | {:,} | `{}` |".format(
                    dataset["dataset"],
                    split,
                    row["users_with_interactions"],
                    row["items_with_interactions"],
                    row["interactions"],
                    row["sha256"],
                )
            )
    lines += [
        "",
        "| Dataset | train/intersection/validation | train/intersection/test | validation/intersection/test | Test raw=verify |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in audit["datasets"]:
        overlap = dataset["split_overlap_interactions"]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                dataset["dataset"],
                overlap["train_validation"],
                overlap["train_test"],
                overlap["validation_test"],
                "PASS" if dataset["assertions"]["raw_test_equals_verify_test"] else "FAIL",
            )
        )
    lines += [
        "",
        "The graph is built from training data. Validation excludes training positives; test excludes training and validation positives. Test data must not select taxonomy, thresholds, hyperparameters, epochs, or checkpoints.",
        "",
        "## Group support",
        "",
        "| Dataset | Group | Items | % catalog | Validation users/positives | Test users/positives |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in audit["datasets"]:
        for group in ("near_cold", "long_tail", "warm", "strict_cold"):
            row = dataset["groups"]["groups"][group]
            lines.append(
                "| {} | {} | {:,} | {:.4f}% | {:,}/{:,} | {:,}/{:,} |".format(
                    dataset["dataset"],
                    group,
                    row["item_count"],
                    row["percent_of_catalog"],
                    row["validation"]["eligible_users"],
                    row["validation"]["positive_interactions"],
                    row["test"]["eligible_users"],
                    row["test"]["positive_interactions"],
                )
            )
    lines += [
        "",
        "Near-cold (1-5) is a subset of Long-tail (1-10); Warm (>10) is disjoint from Long-tail. Strict-cold (=0) is audit-only and is outside the three evaluation slices.",
        "",
        "## Taxonomy coverage",
        "",
        "| Dataset | Valid mapping IDs | Missing | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for dataset in audit["datasets"]:
        row = dataset["taxonomy_coverage"]
        lines.append(
            "| {} | {:,} | {:,} | {:.4f}% |".format(
                dataset["dataset"],
                row["items_with_valid_taxonomy"],
                row["missing_or_empty_taxonomy_items"],
                row["coverage_percent"],
            )
        )
    lines += [
        "",
        "## Gate G1",
        "",
        "**{}**".format(audit["gate_g1"]["status"]),
    ]
    if audit["gate_g1"]["failures"]:
        lines += ["", "Failures:"] + [
            "- " + failure for failure in audit["gate_g1"]["failures"]
        ]
    return "\n".join(lines) + "\n"


def build_all():
    audit, runtime = build_audit()
    preprocessed = ROOT / "preprocessed"
    write_json(preprocessed / "data_protocol_audit.json", audit)
    write_text(
        preprocessed / "data_protocol_audit.md",
        build_audit_markdown(audit),
    )
    taxonomy_tables = {
        "amazon-book": build_amazon_variants(runtime["amazon-book"]["degrees"]),
        "yelp2018": build_yelp_variants(runtime["yelp2018"]["degrees"]),
    }
    protocol = build_experiment_protocol(audit, taxonomy_tables)
    summary = {
        "gate_g1": audit["gate_g1"]["status"],
        "evaluation_protocol_manifests": {
            dataset: sha256_file(
                ROOT
                / "preprocessed"
                / "evaluation_protocol"
                / "split_seed_42"
                / dataset
                / "manifest.json"
            )
            for dataset in DATASETS
        },
        "taxonomy_variant_manifests": {
            dataset: {
                variant: sha256_file(
                    ROOT
                    / "metadata"
                    / "taxonomy_variants"
                    / dataset
                    / variant
                    / "manifest.json"
                )
                for variant in VARIANT_NAMES.values()
            }
            for dataset in DATASETS
        },
        "experiment_protocol_sha256": sha256_file(
            ROOT / "experiment_protocol_v1.json"
        ),
        "metric_count": len(protocol["metrics"]),
    }
    write_json(preprocessed / "week3_build_summary.json", summary)
    print("Gate G1: {}".format(audit["gate_g1"]["status"]))
    return 0 if audit["gate_g1"]["status"] == "PASS" else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build all Week-3 audit/evaluation/taxonomy artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.build:
        raise SystemExit("Use --build")
    raise SystemExit(build_all())
