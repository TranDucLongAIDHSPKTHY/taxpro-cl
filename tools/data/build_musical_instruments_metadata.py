"""Merge Musical_Instruments metadata (2014/2018/2023) using the
musical-instruments item_list.

Additive companion to prepare_metadata.py: that script's "amazon" domain is
wired to a single global dataset (AMAZON_DATASET_NAME = "amazon-book") via
config_path.py, with no per-dataset parameterization -- running it as-is
for musical-instruments would overwrite metadata/item2category_amazon.json
(Books' locked data). This file avoids that entirely by importing only the
pure, already-category-agnostic pieces (_parse_amazon_record,
_normalize_amazon_record, _store_candidate, _select_candidate, merge_domain,
read_item_mapping, write_json_atomic) and re-implementing just the loop that
prepare_metadata.py hardcodes to Books' file paths -- pointed at
metadata/amazon/{year}/meta_Musical_Instruments.json(l) instead. Zero lines
of prepare_metadata.py are modified.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import AMAZON_CATEGORY_METADATA_SOURCES, MUSICAL_INSTRUMENTS_DATASET_NAME
from tools.data.prepare_metadata import (  # noqa: E402  (reused, not duplicated)
    _normalize_amazon_record,
    _parse_amazon_record,
    _require_file,
    _select_candidate,
    _store_candidate,
    merge_domain,
    read_item_mapping,
    write_json_atomic,
)

LOG_EVERY_LINES = 1_000_000
DATASET_NAME = "musical-instruments"

# File paths come from config_path.py's canonical, download-URL-matched
# source-of-truth (AMAZON_CATEGORY_METADATA_URLS covers the download side);
# only the per-vintage id-field regex is specific to this script.
_SOURCE_PATHS = AMAZON_CATEGORY_METADATA_SOURCES[MUSICAL_INSTRUMENTS_DATASET_NAME]
MUSICAL_INSTRUMENTS_SOURCES = {
    "2014": (_SOURCE_PATHS["2014"], re.compile(rb"['\"]asin['\"]\s*:\s*['\"]([^'\"]+)")),
    "2018": (_SOURCE_PATHS["2018"], re.compile(rb'"asin"\s*:\s*"([^"]+)"')),
    "2023": (_SOURCE_PATHS["2023"], re.compile(rb'"parent_asin"\s*:\s*"([^"]+)"')),
}

OUTPUT_NAMES = {
    "metadata": "merged_metadata_musical_instruments.json",
    "categories": "item2category_musical_instruments.json",
    "decisions": "merge_decisions_musical_instruments.json",
    "summary": "merge_summary_musical_instruments.json",
}


def load_musical_instruments_candidates(target_ids, logger):
    targets = set(target_ids)
    candidates = {}
    for version, (path, id_pattern) in MUSICAL_INSTRUMENTS_SOURCES.items():
        _require_file(path)
        logger.info("Reading MusicalInstruments%s: %s", version, path)
        lines = matched = malformed = 0
        with path.open("rb") as stream:
            for line in stream:
                lines += 1
                match = id_pattern.search(line)
                if match is None:
                    malformed += 1
                    continue
                org_id = match.group(1).decode("utf-8", errors="replace")
                if org_id not in targets:
                    if lines % LOG_EVERY_LINES == 0:
                        logger.info(
                            "MusicalInstruments%s: %d rows, %d matched items",
                            version, lines, matched,
                        )
                    continue
                try:
                    raw = _parse_amazon_record(version, line)
                    record = _normalize_amazon_record(org_id, version, raw)
                except (SyntaxError, ValueError, TypeError) as error:
                    malformed += 1
                    logger.warning(
                        "Skipping invalid MusicalInstruments%s record (%s): %s",
                        version, org_id, error,
                    )
                    continue
                _store_candidate(candidates, org_id, version, record)
                matched += 1
                if lines % LOG_EVERY_LINES == 0:
                    logger.info(
                        "MusicalInstruments%s: %d rows, %d matched items",
                        version, lines, matched,
                    )
        logger.info(
            "Completed MusicalInstruments%s: %d rows, %d matched records, %d unrecognized/invalid rows",
            version, lines, matched, malformed,
        )
    return candidates


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("build_musical_instruments_metadata")

    mapping_path = ROOT / "dataset" / DATASET_NAME / "item_list.txt"
    mapping = read_item_mapping(mapping_path)
    logger.info("Musical-instruments item mapping: %d items from %s", len(mapping), mapping_path)

    candidates = load_musical_instruments_candidates(mapping.keys(), logger)
    merged, categories, decisions, summary = merge_domain("amazon", mapping, candidates)

    output_dir = ROOT / "metadata"
    write_json_atomic(output_dir / OUTPUT_NAMES["metadata"], merged, logger)
    write_json_atomic(output_dir / OUTPUT_NAMES["categories"], categories, logger)
    write_json_atomic(output_dir / OUTPUT_NAMES["decisions"], decisions, logger)
    write_json_atomic(output_dir / OUTPUT_NAMES["summary"], summary, logger)
    logger.info(
        "Musical-instruments metadata coverage: %.4f%% (%d/%d)",
        summary["coverage_percent"], summary["matched_items"], summary["total_items"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
