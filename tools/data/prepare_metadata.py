"""Normalize and merge Amazon/Yelp metadata using LightGCN item mappings."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._shared.downloads import configure_logging
from config_path.config_path import (
    AMAZON_METADATA_SOURCES,
    MERGE_OUTPUT_NAMES,
    PROJECT_ROOT,
    YELP_METADATA_SOURCES,
    amazon_metadata_source_file,
    item_mapping_file,
    metadata_root_for,
    preprocessing_log_path,
    temporary_file,
    yelp_metadata_source_file,
)


AMAZON_SOURCES = {
    "2014": (AMAZON_METADATA_SOURCES["2014"], re.compile(rb"['\"]asin['\"]\s*:\s*['\"]([^'\"]+)")),
    "2018": (AMAZON_METADATA_SOURCES["2018"], re.compile(rb'"asin"\s*:\s*"([^"]+)"')),
    "2023": (AMAZON_METADATA_SOURCES["2023"], re.compile(rb'"parent_asin"\s*:\s*"([^"]+)"')),
}

YELP_SOURCES = YELP_METADATA_SOURCES
OUTPUT_NAMES = MERGE_OUTPUT_NAMES

SUMMARY_YEARS = ("2014", "2018", "2021", "2022", "2023")
LOG_EVERY_LINES = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and merge Amazon/Yelp metadata for TaxPro-CL."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (default: the parent directory of tools).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/metadata).",
    )
    parser.add_argument(
        "--domain",
        choices=("all", "amazon", "yelp"),
        default="all",
        help="Pipeline to execute (default: all).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console log level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Preprocessing log file (default: <root>/preprocessing_logs/prepare_metadata.log).",
    )
    return parser.parse_args()


def read_item_mapping(path: Path) -> Dict[str, int]:
    """Read an org_id -> remap_id mapping and validate uniqueness."""
    mapping: Dict[str, int] = {}
    remap_ids = set()
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip().split()
        if header != ["org_id", "remap_id"]:
            raise ValueError("Invalid item-mapping header at {}: {}".format(path, header))
        for line_number, line in enumerate(stream, start=2):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError("Invalid mapping row at {}:{}".format(path, line_number))
            org_id, remap_text = parts
            remap_id = int(remap_text)
            if org_id in mapping or remap_id in remap_ids:
                raise ValueError("Duplicate mapping at {}:{}".format(path, line_number))
            mapping[org_id] = remap_id
            remap_ids.add(remap_id)
    return mapping


def load_amazon_candidates(
    root: Path,
    target_ids: Iterable[str],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    targets = set(target_ids)
    candidates: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for version, (relative_path, id_pattern) in AMAZON_SOURCES.items():
        path = amazon_metadata_source_file(root, version)
        _require_file(path)
        logger.info("Reading Amazon%s: %s", version, path)
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
                        logger.info("Amazon%s: %d rows, %d matched items", version, lines, matched)
                    continue
                try:
                    raw = _parse_amazon_record(version, line)
                    record = _normalize_amazon_record(org_id, version, raw)
                except (SyntaxError, ValueError, TypeError, json.JSONDecodeError) as error:
                    malformed += 1
                    logger.warning("Skipping invalid Amazon%s record (%s): %s", version, org_id, error)
                    continue
                _store_candidate(candidates, org_id, version, record)
                matched += 1
                if lines % LOG_EVERY_LINES == 0:
                    logger.info("Amazon%s: %d rows, %d matched items", version, lines, matched)
        logger.info(
            "Completed Amazon%s: %d rows, %d matched records, %d unrecognized/invalid rows",
            version,
            lines,
            matched,
            malformed,
        )
    return candidates


def load_yelp_candidates(
    root: Path,
    target_ids: Iterable[str],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    targets = set(target_ids)
    candidates: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for version, relative_path in YELP_SOURCES.items():
        path = yelp_metadata_source_file(root, version)
        _require_file(path)
        logger.info("Reading Yelp%s using the ITEM column: %s", version, path)
        rows = matched = 0
        with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not reader.fieldnames or "ITEM" not in reader.fieldnames:
                raise ValueError("Yelp file is missing the ITEM column: {}".format(path))
            for row in reader:
                rows += 1
                org_id = (row.get("ITEM") or "").strip()
                if org_id in targets:
                    record = _normalize_yelp_record(org_id, version, row)
                    _store_candidate(candidates, org_id, version, record)
                    matched += 1
                if rows % LOG_EVERY_LINES == 0:
                    logger.info("Yelp%s: %d rows, %d matched items", version, rows, matched)
        logger.info("Completed Yelp%s: %d rows, %d matched records", version, rows, matched)
    return candidates


def merge_domain(
    domain: str,
    mapping: Mapping[str, int],
    candidates: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Merge items by taxonomy depth, path count, then newest version."""
    merged: Dict[str, Any] = {}
    item2category: Dict[str, Any] = {}
    decisions: List[Dict[str, Any]] = []
    selected_counts = {year: 0 for year in SUMMARY_YEARS}
    matched_items = 0
    total_depth = 0
    total_paths = 0

    for org_id, remap_id in sorted(mapping.items(), key=lambda item: item[1]):
        by_version = candidates.get(org_id, {})
        versions = sorted(by_version, key=int)
        selected, reason = _select_candidate([by_version[version] for version in versions])

        if selected is None:
            record = _missing_record(domain, org_id, remap_id)
            selected_version: Optional[str] = None
            taxonomy_depth = 0
            num_paths = 0
        else:
            record = dict(selected)
            record["item_id"] = remap_id
            record["status"] = "matched"
            selected_version = str(record["source_version"])
            taxonomy_depth = int(record["taxonomy_depth"])
            num_paths = int(record["num_paths"])
            matched_items += 1
            selected_counts[selected_version] += 1
            total_depth += taxonomy_depth
            total_paths += num_paths

        key = str(remap_id)
        merged[key] = record
        item2category[key] = record["taxonomy_paths"]
        decisions.append(
            {
                "item_id": remap_id,
                "org_id": org_id,
                "candidate_versions": versions,
                "selected_version": selected_version,
                "taxonomy_depth": taxonomy_depth,
                "num_paths": num_paths,
                "reason": reason,
            }
        )

    total_items = len(mapping)
    missing_items = total_items - matched_items
    coverage = matched_items / total_items if total_items else 0.0
    summary: Dict[str, Any] = {
        "domain": domain,
        "total_items": total_items,
        "coverage": round(coverage, 8),
        "coverage_percent": round(coverage * 100.0, 4),
        "matched_items": matched_items,
        "missing_items": missing_items,
    }
    for year in SUMMARY_YEARS:
        summary["selected_from_{}".format(year)] = selected_counts[year]
    summary["average_taxonomy_depth"] = round(total_depth / matched_items, 6) if matched_items else 0.0
    summary["average_num_paths"] = round(total_paths / matched_items, 6) if matched_items else 0.0
    return merged, item2category, decisions, summary


def run_domain(domain: str, root: Path, output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    if domain == "amazon":
        mapping_path = item_mapping_file(root, "amazon")
        mapping = read_item_mapping(mapping_path)
        logger.info("Amazon mapping: %d items from %s", len(mapping), mapping_path)
        candidates = load_amazon_candidates(root, mapping.keys(), logger)
    elif domain == "yelp":
        mapping_path = item_mapping_file(root, "yelp")
        mapping = read_item_mapping(mapping_path)
        logger.info("Yelp mapping: %d items from %s", len(mapping), mapping_path)
        candidates = load_yelp_candidates(root, mapping.keys(), logger)
    else:
        raise ValueError("Unsupported domain: {}".format(domain))

    merged, categories, decisions, summary = merge_domain(domain, mapping, candidates)
    names = OUTPUT_NAMES[domain]
    write_json_atomic(output_dir / names["metadata"], merged, logger)
    write_json_atomic(output_dir / names["categories"], categories, logger)
    write_json_atomic(output_dir / names["decisions"], decisions, logger)
    write_json_atomic(output_dir / names["summary"], summary, logger)
    logger.info(
        "%s coverage: %.4f%% (%d/%d)",
        domain.capitalize(),
        summary["coverage_percent"],
        summary["matched_items"],
        summary["total_items"],
    )
    return summary


def write_json_atomic(path: Path, value: Any, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(path)
    logger.info("Writing JSON: %s", path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    logger.info("Wrote %s (%d bytes)", path, path.stat().st_size)


def main() -> int:
    _configure_utf8_streams()
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or metadata_root_for(root)).expanduser().resolve()
    log_file = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else preprocessing_log_path(root, "prepare_metadata")
    )
    logger = configure_logging(args.log_level, log_file, "prepare_metadata")
    domains = ("amazon", "yelp") if args.domain == "all" else (args.domain,)
    logger.info("========== METADATA PREPARATION STARTED ==========")
    logger.info("Preprocessing log: %s", log_file)
    logger.info("Project root: %s", root)
    logger.info("Output directory: %s", output_dir)
    try:
        summaries = [run_domain(domain, root, output_dir, logger) for domain in domains]
    except Exception as error:
        logger.exception("Metadata merge pipeline failed: %s", error)
        return 1

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    logger.info("========== METADATA PREPARATION FINISHED ==========")
    return 0



def _parse_amazon_record(version: str, line: bytes) -> Mapping[str, Any]:
    text = line.decode("utf-8", errors="replace")
    if version == "2014":
        value = ast.literal_eval(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("Amazon record is not an object")
    return value


def _normalize_amazon_record(
    org_id: str,
    version: str,
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    if version == "2014":
        taxonomy_paths = _normalize_taxonomy_paths(raw.get("categories"))
        brand = raw.get("brand")
        average_rating = raw.get("overall")
        rating_number = raw.get("review_count")
    elif version == "2018":
        taxonomy_paths = _single_amazon_path(raw.get("category"), raw.get("main_cat"))
        brand = raw.get("brand")
        average_rating = raw.get("average_rating")
        rating_number = raw.get("rating_number")
    else:
        taxonomy_paths = _single_amazon_path(raw.get("categories"), raw.get("main_category"))
        brand = raw.get("store")
        average_rating = raw.get("average_rating")
        rating_number = raw.get("rating_number")

    return _standard_record(
        domain="amazon",
        org_id=org_id,
        version=version,
        title=raw.get("title"),
        description=raw.get("description"),
        taxonomy_paths=taxonomy_paths,
        brand=brand,
        price=raw.get("price"),
        average_rating=average_rating,
        rating_number=rating_number,
    )


def _normalize_yelp_record(
    org_id: str,
    version: str,
    row: Mapping[str, Any],
) -> Dict[str, Any]:
    categories = []
    for category in (row.get("CATEGORIES") or "").replace("\\/", "/").split(","):
        category = _clean_text(category)
        if category:
            categories.append([category])
    taxonomy_paths = _deduplicate_paths(categories)
    return _standard_record(
        domain="yelp",
        org_id=org_id,
        version=version,
        title=row.get("ITEM_NAME"),
        description=None,
        taxonomy_paths=taxonomy_paths,
        brand=None,
        price=None,
        average_rating=row.get("ITEM_STARS"),
        rating_number=row.get("ITEM_REVIEW_COUNT"),
        address=row.get("ADDRESS"),
        city=row.get("CITY"),
        state=row.get("STATE"),
        postal_code=row.get("POSTAL_CODE"),
        latitude=row.get("LATITUDE"),
        longitude=row.get("LONGITUDE"),
        is_open=row.get("IS_OPEN"),
    )


def _standard_record(
    *,
    domain: str,
    org_id: str,
    version: str,
    title: Any,
    description: Any,
    taxonomy_paths: List[List[str]],
    brand: Any,
    price: Any,
    average_rating: Any,
    rating_number: Any,
    address: Any = None,
    city: Any = None,
    state: Any = None,
    postal_code: Any = None,
    latitude: Any = None,
    longitude: Any = None,
    is_open: Any = None,
) -> Dict[str, Any]:
    return {
        "item_id": None,
        "org_id": org_id,
        "domain": domain,
        "status": "candidate",
        "source_version": version,
        "title": _clean_text(title),
        "description": _normalize_description(description),
        "taxonomy_paths": taxonomy_paths,
        "taxonomy_depth": max((len(path) for path in taxonomy_paths), default=0),
        "num_paths": len(taxonomy_paths),
        "brand": _clean_text(brand),
        "price": _clean_text(price),
        "average_rating": _to_float(average_rating),
        "rating_number": _to_int(rating_number),
        "address": _clean_text(address),
        "city": _clean_text(city),
        "state": _clean_text(state),
        "postal_code": _clean_text(postal_code),
        "latitude": _to_float(latitude),
        "longitude": _to_float(longitude),
        "is_open": _to_int(is_open),
    }


def _missing_record(domain: str, org_id: str, remap_id: int) -> Dict[str, Any]:
    record = _standard_record(
        domain=domain,
        org_id=org_id,
        version="",
        title=None,
        description=None,
        taxonomy_paths=[],
        brand=None,
        price=None,
        average_rating=None,
        rating_number=None,
    )
    record["item_id"] = remap_id
    record["status"] = "missing"
    record["source_version"] = None
    return record


def _select_candidate(candidates: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    if not candidates:
        return None, "missing"
    if len(candidates) == 1:
        return candidates[0], "only_candidate"

    max_depth = max(int(record["taxonomy_depth"]) for record in candidates)
    depth_pool = [record for record in candidates if int(record["taxonomy_depth"]) == max_depth]
    if len(depth_pool) == 1:
        return depth_pool[0], "deeper_taxonomy"

    max_paths = max(int(record["num_paths"]) for record in depth_pool)
    path_pool = [record for record in depth_pool if int(record["num_paths"]) == max_paths]
    if len(path_pool) == 1:
        return path_pool[0], "more_taxonomy_paths"

    return max(path_pool, key=lambda record: int(record["source_version"])), "newer_metadata"


def _store_candidate(
    candidates: Dict[str, Dict[str, Dict[str, Any]]],
    org_id: str,
    version: str,
    record: Dict[str, Any],
) -> None:
    by_version = candidates.setdefault(org_id, {})
    current = by_version.get(version)
    if current is None or _candidate_quality(record) > _candidate_quality(current):
        by_version[version] = record


def _candidate_quality(record: Mapping[str, Any]) -> Tuple[int, int, int]:
    completeness = sum(
        record.get(key) not in (None, "", [], {})
        for key in ("title", "description", "brand", "price", "average_rating", "rating_number")
    )
    return int(record["taxonomy_depth"]), int(record["num_paths"]), completeness


def _single_amazon_path(value: Any, root_category: Any) -> List[List[str]]:
    if isinstance(value, (list, tuple)):
        path = [_clean_text(part) for part in value]
        path = [part for part in path if part]
    elif value is None:
        path = []
    else:
        cleaned = _clean_text(value)
        path = [cleaned] if cleaned else []
    root = _clean_text(root_category)
    if root and (not path or path[0].casefold() != root.casefold()):
        path.insert(0, root)
    return [path] if path else []


def _normalize_taxonomy_paths(value: Any) -> List[List[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [[cleaned]] if cleaned else []
    if not isinstance(value, (list, tuple)):
        return []
    if all(not isinstance(part, (list, tuple)) for part in value):
        path = [_clean_text(part) for part in value]
        path = [part for part in path if part]
        return [path] if path else []
    paths = []
    for raw_path in value:
        if not isinstance(raw_path, (list, tuple)):
            continue
        path = [_clean_text(part) for part in raw_path]
        path = [part for part in path if part]
        if path:
            paths.append(path)
    return _deduplicate_paths(paths)


def _deduplicate_paths(paths: Iterable[List[str]]) -> List[List[str]]:
    unique: List[List[str]] = []
    seen = set()
    for path in paths:
        key = tuple(part.casefold() for part in path)
        if key and key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _normalize_description(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [_clean_text(part) for part in value]
        parts = [part for part in parts if part]
        return "\n".join(parts) if parts else None
    return _clean_text(value)


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = html.unescape(str(value)).strip()
    return text or None


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError("Required file not found: {}".format(path))


def _configure_utf8_streams() -> None:
    """Ensure Unicode CLI and log output works on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    sys.exit(main())
