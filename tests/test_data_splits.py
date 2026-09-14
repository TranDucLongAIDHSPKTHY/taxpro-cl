import tempfile
import unittest
from pathlib import Path

from tools.data.build_splits import (
    ALGORITHM_ID,
    build_dataset,
    iterative_k_core,
    split_train_validation,
)


class DataSplitTests(unittest.TestCase):
    def test_iterative_k_core_runs_until_both_sides_are_stable(self):
        pairs = {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 1),
            (2, 2),
            (3, 2),
        }
        filtered, statistics = iterative_k_core(pairs, minimum_degree=2)
        self.assertEqual(filtered, {(0, 0), (0, 1), (1, 0), (1, 1)})
        self.assertGreaterEqual(len(statistics["iterations"]), 3)
        self.assertEqual(statistics["minimum_user_degree"], 2)
        self.assertEqual(statistics["minimum_item_degree"], 2)

    def test_per_user_split_is_deterministic_and_disjoint(self):
        pairs = {
            (user, item)
            for user, degree in ((0, 1), (1, 5), (2, 15), (3, 25))
            for item in range(user * 100, user * 100 + degree)
        }
        first = split_train_validation(pairs, validation_ratio=0.1, split_seed=42)
        second = split_train_validation(pairs, validation_ratio=0.1, split_seed=42)
        different = split_train_validation(
            pairs, validation_ratio=0.1, split_seed=123
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        train, validation = first
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, pairs)
        self.assertEqual(
            [sum(user == target for user, _ in validation) for target in range(4)],
            [0, 1, 1, 2],
        )
        with self.assertRaises(ValueError):
            split_train_validation(pairs, split_seed=43)
        with self.assertRaises(ValueError):
            split_train_validation(pairs, validation_ratio=0.2)

    def test_build_dataset_preserves_test_bytes_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "toy"
            output = root / "output"
            source.mkdir(parents=True)
            (source / "train.txt").write_text(
                "0 0 1 2\n1 0 1 2\n", encoding="utf-8", newline="\n"
            )
            test_content = "0 3\r\n1 4\r\n2\r\n"
            (source / "test.txt").write_bytes(test_content.encode("ascii"))

            manifest = build_dataset(
                "toy",
                root / "source",
                output,
                minimum_degree=2,
                validation_ratio=0.1,
                split_seed=42,
            )

            self.assertEqual(manifest["algorithm"], ALGORITHM_ID)
            self.assertEqual(manifest["counts"]["five_core"], 6)
            self.assertEqual(manifest["counts"]["train"], 4)
            self.assertEqual(manifest["counts"]["validation"], 2)
            self.assertTrue(
                manifest["assertions"]["test_preserved_byte_for_byte"]
            )
            self.assertEqual(
                (output / "toy" / "test.txt").read_bytes(),
                test_content.encode("ascii"),
            )
            self.assertTrue((output / "toy" / "split_manifest.json").is_file())

            compared = build_dataset(
                "toy",
                root / "source",
                root / "compared",
                minimum_degree=2,
                validation_ratio=0.1,
                split_seed=42,
                reference_root=output,
            )
            self.assertTrue(compared["locked_reference"]["all_outputs_match"])

            with self.assertRaises(FileExistsError):
                build_dataset(
                    "toy",
                    root / "source",
                    output,
                    minimum_degree=2,
                )


if __name__ == "__main__":
    unittest.main()
