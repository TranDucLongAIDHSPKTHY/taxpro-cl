"""Revalidate the standalone selective migration deterministically."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from config_path.config_path import PROJECT_ROOT


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest_path = ROOT / "manifests" / "migration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    checked = 0
    for entry in manifest["entries"]:
        destination = ROOT / Path(entry["destination_path"])
        checked += 1
        if not destination.is_file():
            failures.append("missing:" + entry["destination_path"])
            continue
        if destination.stat().st_size != int(entry["size"]):
            failures.append("size:" + entry["destination_path"])
        if sha256_file(destination) != entry["sha256"]:
            failures.append("destination_hash:" + entry["destination_path"])
        if entry["copied_or_generated"] == "copied":
            source = SOURCE_ROOT / Path(entry["source_path"])
            if not source.is_file() or sha256_file(source) != entry["source_sha256"]:
                failures.append("source_hash:" + entry["destination_path"])
    if PROJECT_ROOT.resolve() != ROOT.resolve():
        failures.append("project_root_not_portable")
    if (ROOT / ".git").exists():
        failures.append("nested_git_repository")
    forbidden = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if "__pycache__" in path.parts
        or path.suffix == ".pyc"
        or path.name.endswith(".tmp")
    ]
    if forbidden:
        failures.extend("forbidden:" + value for value in forbidden)
    parent_literal = str(SOURCE_ROOT).lower()
    runtime_hits = []
    for directory in ("config_path", "models", "utility", "tools"):
        for path in (ROOT / directory).rglob("*.py"):
            if parent_literal in path.read_text(encoding="utf-8").lower():
                runtime_hits.append(path.relative_to(ROOT).as_posix())
    if runtime_hits:
        failures.extend("parent_runtime_path:" + value for value in runtime_hits)
    payload = {
        "protocol_version": "taxprocl-p0-week5-v1",
        "checked_files": checked,
        "forbidden_artifacts": forbidden,
        "runtime_parent_path_hits": runtime_hits,
        "failures": sorted(failures),
        "status": "PASS" if not failures else "FAIL",
    }
    output = ROOT / "manifests" / "migration_validation.json"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
