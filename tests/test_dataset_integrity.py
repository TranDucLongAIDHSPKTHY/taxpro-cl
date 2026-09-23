import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


DATASETS = ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing")
TAXONOMY_POLICIES = ("no_merge", "merge_t5", "merge_t10", "merge_t15")


class DatasetIntegrityTests(unittest.TestCase):
    def test_dataset_and_evaluation_hashes_match_local_manifests(self):
        for dataset in DATASETS:
            directory = (
                ROOT
                / "preprocessed"
                / "evaluation_protocol"
                / "split_seed_42"
                / dataset
            )
            if not (directory / "manifest.json").is_file():
                self.skipTest("Protocol artifacts have not been rebuilt")
            manifest = json.loads((directory / "manifest.json").read_text())
            for split in ("train", "validation", "test"):
                actual = sha256_file(
                    ROOT / "dataset_verify" / dataset / (split + ".txt")
                )
                self.assertEqual(actual, manifest["source_hashes"][split])
            self.assertEqual(
                sha256_file(directory / "group_masks.npz"),
                manifest["output_hashes"]["group_masks.npz"],
            )

    def test_taxonomy_variant_outputs_match_their_manifests(self):
        checked = 0
        for dataset in DATASETS:
            for policy in TAXONOMY_POLICIES:
                directory = ROOT / "metadata" / "taxonomy_variants" / dataset / policy
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    self.skipTest("Taxonomy variants have not been rebuilt")
                manifest = json.loads(manifest_path.read_text())
                for name, expected in manifest["output_hashes"].items():
                    self.assertEqual(
                        sha256_file(directory / name), expected, msg=str(directory / name)
                    )
                checked += 1
        self.assertEqual(checked, len(DATASETS) * len(TAXONOMY_POLICIES))

    def test_tracked_tree_has_no_nested_git_or_runtime_cache(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=str(ROOT)
        ).decode("utf-8").split("\0")
        forbidden = [
            path
            for path in tracked
            if ".git" in Path(path).parts
            or "__pycache__" in Path(path).parts
            or Path(path).suffix == ".pyc"
        ]
        self.assertEqual(forbidden, [])
