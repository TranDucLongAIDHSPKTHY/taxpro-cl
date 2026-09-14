"""Export small, verifiable run evidence without copying model checkpoints."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


INCLUDED_NAMES = {
    "config_resolved.json",
    "environment.json",
    "run_manifest.json",
    "training.log",
    "validation_metrics.json",
    "final_test_metrics.json",
    "final_test_group_metrics.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(run_dir, output_dir):
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest = run_dir / "run_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Missing run_manifest.json: {}".format(run_dir))
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if payload.get("status") != "completed":
        raise ValueError("Only completed runs can be exported")
    configuration = json.dumps(
        payload.get("configuration", {}), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config_id = hashlib.sha256(configuration).hexdigest()[:12]
    run_id = "{}-{}-seed{}-{}".format(
        payload.get("model", "model"),
        payload.get("dataset", "dataset"),
        payload.get("training_seed", "unknown"),
        config_id,
    ).lower()
    target = output_dir / run_id
    if target.exists():
        raise FileExistsError("Evidence bundle already exists: {}".format(target))
    target.mkdir(parents=True)
    copied = []
    for source in sorted(run_dir.rglob("*")):
        if not source.is_file() or source.name not in INCLUDED_NAMES:
            continue
        relative = source.relative_to(run_dir)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({"path": relative.as_posix(), "sha256": sha256(destination)})
    if not any(item["path"] == "run_manifest.json" for item in copied):
        raise RuntimeError("Evidence export did not include the run manifest")
    (target / "checksums.json").write_text(
        json.dumps({"schema_version": 1, "files": copied}, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.run_dir, args.output_dir))


if __name__ == "__main__":
    main()
