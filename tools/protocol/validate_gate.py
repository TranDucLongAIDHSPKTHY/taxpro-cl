"""Validate the complete Week-3 Gate G1/G1.5 handoff."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_TEST_MODULES = (
    "tests.test_data_splits",
    "tests.test_dataset_integrity",
    "tests.test_group_evaluator",
    "tests.test_runner_result_contract",
    "tests.test_taxpro_runner",
    "tests.test_tools_layout",
    "tests.test_ablation_compiler",
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main():
    audit = load(ROOT / "preprocessed" / "data_protocol_audit.json")
    determinism = load(
        ROOT / "preprocessed" / "week3_determinism_validation.json"
    )
    experiment = load(ROOT / "experiment_protocol_v1.json")
    evaluation_manifests = list(
        (ROOT / "preprocessed" / "evaluation_protocol" / "split_seed_42").glob(
            "*/manifest.json"
        )
    )
    taxonomy_manifests = list(
        (ROOT / "metadata" / "taxonomy_variants").glob("*/*/manifest.json")
    )
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
        ]
        + list(PROTOCOL_TEST_MODULES),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    checks = {
        "gate_g1_pass": audit["gate_g1"]["status"] == "PASS",
        "test_hashes_invariant": (
            audit["test_hashes_before"] == audit["test_hashes_after"]
        ),
        "zero_split_leakage": all(
            all(value == 0 for value in row["split_overlap_interactions"].values())
            for row in audit["datasets"]
        ),
        "evaluation_manifests_count_4": len(evaluation_manifests) == 4,
        "taxonomy_manifests_count_16": len(taxonomy_manifests) == 16,
        "metric_count_24": len(experiment["metrics"]) == 24,
        "unit_protocol_tests_pass": tests.returncode == 0,
        "determinism_pass": determinism["status"] == "PASS",
        "determinism_mismatch_count_0": not determinism["mismatches"],
        "generic_evaluator_exists": (
            ROOT / "utility" / "utility_train" / "group_evaluator.py"
        ).is_file(),
        "batch_test_integration_exists": (
            "groupwise"
            in (ROOT / "utility" / "utility_train" / "batch_test.py").read_text(
                encoding="utf-8"
            )
        ),
    }
    result = {
        "gate_g1": "PASS" if checks["gate_g1_pass"] else "FAIL",
        "gate_g1_5": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "unit_test_return_code": tests.returncode,
        "unit_test_summary": (
            tests.stderr.strip().splitlines()[-4:]
            if tests.stderr.strip()
            else tests.stdout.strip().splitlines()[-4:]
        ),
        "unit_test_scope": list(PROTOCOL_TEST_MODULES),
        "optional_checkpoint_smoke": {
            "status": "NOT_RUN",
            "required": False,
            "reason": (
                "No checkpoint could be validated with torch.load because the "
                "Week-3 shell has no PyTorch runtime."
            ),
        },
        "scope": "No Week-4 baseline training and no TaxProCL full training.",
    }
    write(ROOT / "preprocessed" / "gate_g1_5_validation.json", result)
    print("Gate G1.5: {}".format(result["gate_g1_5"]))
    return 0 if result["gate_g1_5"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
