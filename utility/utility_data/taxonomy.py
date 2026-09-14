"""Deterministic loader for the train-locked Week-3 taxonomy variants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config_path.config_path import (
    evaluation_protocol_dir,
    taxonomy_variant_dir,
    verified_dataset_dir,
)


SUPPORTED_POLICIES = ("no_merge", "merge_t5", "merge_t10", "merge_t15")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


@dataclass(frozen=True)
class TaxonomyArtifacts:
    dataset: str
    policy: str
    granularity: str
    directory: Path
    taxonomy_hash: str
    manifest: dict
    item_to_leaf_id: np.ndarray
    valid_taxonomy_mask: np.ndarray
    train_observed_mask: np.ndarray
    valid_train_mask: np.ndarray
    leaf_id_to_name: dict
    provenance_path: Path
    statistics: dict

    @property
    def num_items(self):
        return int(self.item_to_leaf_id.shape[0])

    @property
    def num_prototypes(self):
        valid = self.item_to_leaf_id[self.valid_train_mask]
        return int(valid.max()) + 1 if valid.size else 0

    def as_torch(self, device=None):
        import torch

        return {
            "item_to_leaf_id": torch.as_tensor(
                self.item_to_leaf_id, dtype=torch.long, device=device
            ),
            "valid_taxonomy_mask": torch.as_tensor(
                self.valid_taxonomy_mask, dtype=torch.bool, device=device
            ),
            "train_observed_mask": torch.as_tensor(
                self.train_observed_mask, dtype=torch.bool, device=device
            ),
            "valid_train_mask": torch.as_tensor(
                self.valid_train_mask, dtype=torch.bool, device=device
            ),
        }


def validate_variant_manifest(dataset, policy, directory, manifest):
    if dataset not in {
        "amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing",
    }:
        raise ValueError("Unsupported dataset: {}".format(dataset))
    if policy not in SUPPORTED_POLICIES:
        raise ValueError("Unsupported taxonomy policy: {}".format(policy))
    if manifest.get("dataset") != dataset or manifest.get("variant") != policy:
        raise ValueError("Taxonomy manifest dataset/policy mismatch")
    if manifest.get("protocol_version") != "taxprocl-p0-week3-v1":
        raise ValueError("Unsupported taxonomy protocol version")
    if manifest.get("policy", {}).get("validation_or_test_used") is not False:
        raise ValueError("Taxonomy manifest is not train-only")
    policy_payload = manifest.get("policy", {})
    explicit_shared = policy_payload.get("shared_other_or_unknown_prototype")
    invalid_policy = str(policy_payload.get("invalid_policy", "")).lower()
    if explicit_shared is True or (
        explicit_shared is None and "exclude" not in invalid_policy
    ):
        raise ValueError("Shared unknown prototype is not supported")
    required = {
        "item_to_leaf.json",
        "item_to_leaf_id.npy",
        "valid_taxonomy_mask.npy",
        "leaf_id_to_name.json",
        "mapping_provenance.jsonl",
        "taxonomy_statistics.json",
    }
    hashes = manifest.get("output_hashes", {})
    if set(hashes) != required:
        raise ValueError("Taxonomy manifest output set is incomplete")
    for name in sorted(required):
        path = Path(directory) / name
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Taxonomy artifact hash mismatch: {}".format(name))
    train_path = verified_dataset_dir(dataset) / "train.txt"
    if sha256_file(train_path) != manifest.get("source_hashes", {}).get("train"):
        raise ValueError("Taxonomy train source hash mismatch")


def _derive_parent_taxonomy(item_to_leaf, valid_mask, leaf_names):
    """Collapse Amazon leaf labels to their immediate parent deterministically."""
    parent_names = {}
    item_to_parent = np.full(item_to_leaf.shape, -1, dtype=np.int64)
    valid_leaf_ids = sorted(
        {int(value) for value in item_to_leaf[valid_mask] if int(value) >= 0}
    )
    for leaf_id in valid_leaf_ids:
        leaf_name = leaf_names.get(str(leaf_id), leaf_names.get(leaf_id))
        if not leaf_name:
            raise ValueError("Missing name for valid leaf id {}".format(leaf_id))
        parts = [part.strip() for part in str(leaf_name).split(">") if part.strip()]
        parent_name = " > ".join(parts[:-1]) if len(parts) > 1 else parts[0]
        parent_names.setdefault(parent_name, len(parent_names))
        item_to_parent[item_to_leaf == leaf_id] = parent_names[parent_name]
    id_to_name = {str(value): key for key, value in parent_names.items()}
    return item_to_parent, id_to_name


def _derived_taxonomy_hash(source_hash, granularity, item_to_group, group_names):
    digest = hashlib.sha256()
    digest.update(str(source_hash).encode("utf-8"))
    digest.update(str(granularity).encode("utf-8"))
    digest.update(np.ascontiguousarray(item_to_group).tobytes())
    digest.update(
        json.dumps(group_names, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def load_taxonomy(dataset, policy="no_merge", granularity="leaf"):
    if granularity not in {"leaf", "parent"}:
        raise ValueError("Unsupported taxonomy granularity: {}".format(granularity))
    if granularity == "parent" and dataset != "amazon-book":
        raise ValueError("Parent-level ablation A5 is available only for amazon-book")
    directory = taxonomy_variant_dir(dataset, policy)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing taxonomy manifest: {}".format(manifest_path))
    manifest = read_json(manifest_path)
    validate_variant_manifest(dataset, policy, directory, manifest)

    item_to_leaf_id = np.load(directory / "item_to_leaf_id.npy", allow_pickle=False)
    valid_taxonomy_mask = np.load(
        directory / "valid_taxonomy_mask.npy", allow_pickle=False
    ).astype(bool, copy=False)
    degree_payload = read_json(
        evaluation_protocol_dir(dataset) / "item_degrees_train.json"
    )
    degrees = np.asarray(degree_payload["degrees"], dtype=np.int64)
    if not (
        item_to_leaf_id.ndim == 1
        and valid_taxonomy_mask.ndim == 1
        and degrees.ndim == 1
        and len(item_to_leaf_id) == len(valid_taxonomy_mask) == len(degrees)
    ):
        raise ValueError("Taxonomy/degree arrays do not cover the same catalog")
    if np.any(valid_taxonomy_mask & (item_to_leaf_id < 0)):
        raise ValueError("Valid taxonomy item has a negative leaf id")
    train_observed_mask = degrees > 0
    valid_train_mask = (
        valid_taxonomy_mask & train_observed_mask & (item_to_leaf_id >= 0)
    )
    statistics = read_json(directory / "taxonomy_statistics.json")
    if int(statistics["catalog_items"]) != len(item_to_leaf_id):
        raise ValueError("Taxonomy statistics catalog size mismatch")
    if int(valid_taxonomy_mask.sum()) + int(statistics["invalid_items"]) != len(
        item_to_leaf_id
    ):
        raise ValueError("Taxonomy valid/invalid accounting mismatch")

    leaf_id_to_name = read_json(directory / "leaf_id_to_name.json")
    taxonomy_hash = sha256_file(manifest_path)
    if granularity == "parent":
        item_to_leaf_id, leaf_id_to_name = _derive_parent_taxonomy(
            item_to_leaf_id, valid_taxonomy_mask, leaf_id_to_name
        )
        taxonomy_hash = _derived_taxonomy_hash(
            taxonomy_hash, granularity, item_to_leaf_id, leaf_id_to_name
        )
        statistics = dict(statistics)
        statistics["derived_granularity"] = "parent"
        statistics["derived_parent_count"] = len(leaf_id_to_name)

    return TaxonomyArtifacts(
        dataset=dataset,
        policy=policy,
        granularity=granularity,
        directory=directory,
        taxonomy_hash=taxonomy_hash,
        manifest=manifest,
        item_to_leaf_id=item_to_leaf_id.astype(np.int64, copy=False),
        valid_taxonomy_mask=valid_taxonomy_mask,
        train_observed_mask=train_observed_mask,
        valid_train_mask=valid_train_mask,
        leaf_id_to_name=leaf_id_to_name,
        provenance_path=directory / "mapping_provenance.jsonl",
        statistics=statistics,
    )
