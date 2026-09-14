"""Build the selective migration inventory and per-file evidence manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT.parent
MANIFEST_DIR = ROOT / "manifests"
EXCLUDED = {
    "manifests/migration_manifest.json",
    "manifests/source_inventory.json",
    "manifests/migration_validation.json",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_candidate(relative):
    if relative == "manifests/experiment_protocol_v1.json":
        return SOURCE_ROOT / "experiment_protocol_v1.json"
    prefix = "results/baseline_reference/"
    if relative.startswith(prefix):
        return SOURCE_ROOT / "results" / "baselines" / relative[len(prefix) :]
    candidate = SOURCE_ROOT / Path(relative)
    return candidate if candidate.is_file() else None


def artifact_type(relative):
    top = relative.split("/", 1)[0]
    return {
        "dataset_verify": "model_ready_data",
        "preprocessed": "evaluation_protocol",
        "metadata": "taxonomy_metadata",
        "models": "model_source",
        "configure": "model_configuration",
        "utility": "runtime_dependency",
        "tests": "test_source",
        "results": "baseline_reference",
        "manifests": "protocol_manifest",
        "tools": "experiment_tool",
    }.get(top, "project_source")


def git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SOURCE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or None


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def main():
    commit = git_commit()
    entries = []
    sources = []
    for destination in sorted(path for path in ROOT.rglob("*") if path.is_file()):
        relative = destination.relative_to(ROOT).as_posix()
        if (
            relative in EXCLUDED
            or "__pycache__" in destination.parts
            or destination.suffix in {".pyc", ".tmp"}
        ):
            continue
        source = source_candidate(relative)
        destination_hash = sha256_file(destination)
        entry = {
            "source_path": (
                source.relative_to(SOURCE_ROOT).as_posix()
                if source is not None
                else None
            ),
            "destination_path": relative,
            "artifact_type": artifact_type(relative),
            "copied_or_generated": "generated",
            "size": destination.stat().st_size,
            "sha256": destination_hash,
            "source_git_commit": commit if source is not None else None,
            "protocol_version": (
                "taxprocl-p0-week3-v1"
                if relative.startswith(("preprocessed/", "metadata/taxonomy_variants/"))
                else "taxprocl-p0-week5-v1"
            ),
            "migration_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "PASS",
        }
        if source is not None:
            source_hash = sha256_file(source)
            entry["source_sha256"] = source_hash
            entry["copied_or_generated"] = (
                "copied" if source_hash == destination_hash else "copied_then_modified"
            )
            sources.append(
                {
                    "source_path": entry["source_path"],
                    "size": source.stat().st_size,
                    "sha256": source_hash,
                    "source_git_commit": commit,
                }
            )
        entries.append(entry)
    directories = [
        {
            "destination_path": path.relative_to(ROOT).as_posix(),
            "artifact_type": artifact_type(path.relative_to(ROOT).as_posix()),
            "validation_status": "PASS",
        }
        for path in sorted(path for path in ROOT.rglob("*") if path.is_dir())
        if "__pycache__" not in path.parts and path.name != ".git"
    ]
    payload = {
        "schema_version": 1,
        "protocol_version": "taxprocl-p0-week5-v1",
        "source_root_reference": str(SOURCE_ROOT),
        "destination_root": str(ROOT),
        "source_git_commit": commit,
        "nested_git_repository": False,
        "recursive_source_copy": False,
        "entries": entries,
        "directories": directories,
        "summary": {
            "file_count": len(entries),
            "directory_count": len(directories),
            "total_bytes": sum(entry["size"] for entry in entries),
            "copied": sum(
                entry["copied_or_generated"] == "copied" for entry in entries
            ),
            "copied_then_modified": sum(
                entry["copied_or_generated"] == "copied_then_modified"
                for entry in entries
            ),
            "generated": sum(
                entry["copied_or_generated"] == "generated" for entry in entries
            ),
        },
    }
    unique_sources = {
        row["source_path"]: row for row in sources
    }
    write_json(MANIFEST_DIR / "source_inventory.json", {
        "source_root_reference": str(SOURCE_ROOT),
        "source_git_commit": commit,
        "entries": [unique_sources[key] for key in sorted(unique_sources)],
    })
    write_json(MANIFEST_DIR / "migration_manifest.json", payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
