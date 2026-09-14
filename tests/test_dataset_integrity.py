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


class DatasetIntegrityTests(unittest.TestCase):
    def test_dataset_and_evaluation_hashes_match_local_manifests(self):
        for dataset in ("amazon-book", "yelp2018"):
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

    def test_migration_manifest_validates_all_copied_data(self):
        manifest_path = ROOT / "manifests" / "migration_manifest.json"
        if not manifest_path.is_file():
            self.skipTest("Migration manifest has not been rebuilt")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        migrated_data = [
            entry
            for entry in manifest["entries"]
            if entry["destination_path"].startswith(
                ("dataset_verify/", "preprocessed/", "metadata/taxonomy_variants/")
            )
        ]
        self.assertGreater(len(migrated_data), 0)
        for entry in migrated_data:
            self.assertEqual(entry["copied_or_generated"], "copied")
            self.assertEqual(
                sha256_file(ROOT / entry["destination_path"]), entry["sha256"]
            )
