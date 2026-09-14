"""Rerun the Week-3 builder and prove byte-level output determinism."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.protocol.build import sha256_file, write_json  # noqa: E402


def artifact_files():
    files = [
        ROOT / "preprocessed" / "data_protocol_audit.json",
        ROOT / "preprocessed" / "data_protocol_audit.md",
        ROOT / "preprocessed" / "week3_build_summary.json",
        ROOT / "experiment_protocol_v1.json",
    ]
    for manifest_path in sorted(
        (ROOT / "preprocessed" / "evaluation_protocol").glob(
            "split_seed_42/*/manifest.json"
        )
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files.append(manifest_path)
        files.extend(manifest_path.parent / name for name in manifest["output_hashes"])
    for manifest_path in sorted(
        (ROOT / "metadata" / "taxonomy_variants").glob("*/*/manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files.append(manifest_path)
        files.extend(manifest_path.parent / name for name in manifest["output_hashes"])
    return sorted(set(files))


def snapshot():
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in artifact_files()
    }


def main():
    before = snapshot()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.protocol.build",
            "--build",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    after = snapshot()
    mismatches = {
        path: {"before": before.get(path), "after": after.get(path)}
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    }
    audit = json.loads(
        (ROOT / "preprocessed" / "data_protocol_audit.json").read_text(
            encoding="utf-8"
        )
    )
    result = {
        "status": (
            "PASS"
            if completed.returncode == 0
            and not mismatches
            and audit["test_hashes_before"] == audit["test_hashes_after"]
            else "FAIL"
        ),
        "builder_return_code": completed.returncode,
        "builder_stdout": completed.stdout.strip(),
        "builder_stderr": completed.stderr.strip(),
        "compared_files": len(before),
        "mismatches": mismatches,
        "test_hashes_invariant": (
            audit["test_hashes_before"] == audit["test_hashes_after"]
        ),
    }
    write_json(
        ROOT / "preprocessed" / "week3_determinism_validation.json",
        result,
    )
    print("Determinism: {}".format(result["status"]))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
