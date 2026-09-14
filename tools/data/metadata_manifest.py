"""Inventory extracted metadata without retaining downloaded archives."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import AMAZON_METADATA_URLS, METADATA_DIR, YELP_METADATA_URLS


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(hash_content=False):
    entries = []
    sources = {
        **{"amazon-{}".format(year): url for year, url in AMAZON_METADATA_URLS.items()},
        **{"yelp-{}".format(year): url for year, url in YELP_METADATA_URLS.items()},
    }
    roots = [METADATA_DIR / "amazon", METADATA_DIR / "yelp"]
    for source_root in roots:
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() in {".gz", ".zip"}:
                continue
            item = {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
            }
            if hash_content:
                item["sha256"] = sha256(path)
            entries.append(item)
    return {
        "schema_version": 1,
        "policy": "keep-extracted-delete-download-archives",
        "sources": sources,
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-content", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "manifests" / "metadata_sources.json"
    )
    args = parser.parse_args()
    payload = build(args.hash_content)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
