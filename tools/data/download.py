"""Download the LightGCN datasets and Amazon/Yelp metadata for TaxPro-CL."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._shared.downloads import (
    configure_logging,
    download_file,
    extract_gzip,
    extract_zip,
)
from config_path.config_path import (
    AMAZON_METADATA_URLS,
    DATASET_FILES,
    DATASET_REMOTE_FILES,
    LIGHTGCN_RAW_URL,
    LIGHTGCN_YELP_SOURCE_URL,
    PROJECT_ROOT,
    README_FILE_NAME,
    YELP_DATASET_NAME,
    YELP_METADATA_URLS,
    amazon_metadata_year_dir,
    dataset_directory,
    metadata_root_for,
    preprocessing_log_path,
    yelp_metadata_archive,
    yelp_metadata_root,
    yelp_metadata_year_dir,
)

YELP_README = """# Yelp2018

Processed LightGCN data downloaded from:
{source_url}

The files `train.txt`, `test.txt`, `item_list.txt`, and `user_list.txt` retain
the original LightGCN names and structure.
""".format(source_url=LIGHTGCN_YELP_SOURCE_URL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LightGCN datasets and Amazon/Yelp metadata."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and extract again even when outputs already exist.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep .gz/.zip files after extraction (default: delete duplicates).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (default: the parent directory of tools).",
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
        help="Preprocessing log file (default: <root>/preprocessing_logs/preprocessing.log).",
    )
    return parser.parse_args()


def download_datasets(root: Path, force: bool, logger: logging.Logger) -> None:
    logger.info("Starting LightGCN dataset download")
    for dataset, remote_files in DATASET_REMOTE_FILES.items():
        dataset_dir = dataset_directory(root, dataset)
        for filename in remote_files:
            download_file(
                "{}/{}/{}".format(LIGHTGCN_RAW_URL, dataset, filename),
                dataset_dir / filename,
                force=force,
                logger=logger,
            )
        if dataset == YELP_DATASET_NAME:
            _write_yelp_readme(dataset_dir / README_FILE_NAME, force, logger)
    _require_files(root, _dataset_outputs(root), "LightGCN dataset")
    logger.info("LightGCN datasets are complete")


def download_metadata(
    root: Path, force: bool, keep_archives: bool, logger: logging.Logger
) -> None:
    logger.info("Starting Amazon metadata download")
    for year, url in AMAZON_METADATA_URLS.items():
        year_dir = amazon_metadata_year_dir(root, year)
        archive = year_dir / Path(url).name
        output = year_dir / archive.name[:-3]
        if output.is_file() and output.stat().st_size > 0 and not force:
            logger.info("Skipping metadata that is already extracted: %s", output)
        else:
            download_file(url, archive, force=force, logger=logger)
            extract_gzip(archive, output, force=force, logger=logger)
        _require_nonempty_file(output)
        _remove_archive(archive, keep_archives, logger)

    logger.info("Starting Yelp metadata download")
    yelp_root = yelp_metadata_root(root)
    for year, url in YELP_METADATA_URLS.items():
        archive = yelp_metadata_archive(root, year)
        output_dir = yelp_metadata_year_dir(root, year)
        extracted = output_dir.is_dir() and any(
            path.is_file() and path.stat().st_size > 0
            for path in output_dir.rglob("*")
        )
        if extracted and not force:
            logger.info("Skipping metadata that is already extracted: %s", output_dir)
        else:
            download_file(url, archive, force=force, logger=logger)
            extract_zip(archive, output_dir, force=force, logger=logger)
        if not output_dir.is_dir() or not any(
            path.is_file() and path.stat().st_size > 0
            for path in output_dir.rglob("*")
        ):
            raise RuntimeError("Yelp {} metadata was not extracted: {}".format(year, output_dir))
        _remove_archive(archive, keep_archives, logger)

    _require_files(root, _amazon_outputs(root), "Amazon metadata")
    for year in YELP_METADATA_URLS:
        output_dir = yelp_metadata_year_dir(root, year)
        if not output_dir.is_dir() or not any(path.is_file() for path in output_dir.rglob("*")):
            raise RuntimeError("Yelp {} metadata was not extracted: {}".format(year, output_dir))
    logger.info("Amazon and Yelp metadata are downloaded and extracted")


def main() -> int:
    _configure_utf8_streams()
    args = parse_args()
    root = args.root.expanduser().resolve()
    log_file = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else preprocessing_log_path(root, "pipeline")
    )
    logger = configure_logging(args.log_level, log_file, "download_data")
    logger.info("========== DATA DOWNLOAD STARTED ==========")
    logger.info("Preprocessing log: %s", log_file)
    logger.info("Project root: %s", root)
    logger.info("Force mode: %s", "enabled" if args.force else "disabled")
    try:
        download_datasets(root, args.force, logger)
        download_metadata(root, args.force, args.keep_archives, logger)
    except Exception as error:
        logger.exception("Data preparation failed: %s", error)
        return 1
    logger.info("Data preparation completed")
    logger.info("========== DATA DOWNLOAD FINISHED ==========")
    return 0


def _write_yelp_readme(path: Path, force: bool, logger: logging.Logger) -> None:
    if path.is_file() and not force:
        logger.info("Skipping existing file: %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(YELP_README, encoding="utf-8")
    logger.info("Created Yelp2018 README: %s", path)


def _dataset_outputs(root: Path) -> Iterable[Path]:
    for dataset in DATASET_REMOTE_FILES:
        for filename in DATASET_FILES:
            yield dataset_directory(root, dataset) / filename


def _amazon_outputs(root: Path) -> Iterable[Path]:
    for year, url in AMAZON_METADATA_URLS.items():
        yield amazon_metadata_year_dir(root, year) / Path(url).name[:-3]


def _require_files(root: Path, relative_paths: Iterable[Path], label: str) -> None:
    missing = []
    for path in relative_paths:
        resolved = root / path
        if not resolved.is_file():
            try:
                display_path = resolved.relative_to(root)
            except ValueError:
                display_path = resolved
            missing.append(str(display_path))
    if missing:
        raise RuntimeError("Missing {} files: {}".format(label, ", ".join(missing)))


def _require_nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("Invalid extracted file: {}".format(path))


def _remove_archive(path: Path, keep: bool, logger: logging.Logger) -> None:
    if keep or not path.exists():
        return
    path.unlink()
    logger.info("Deleted archive after validating extracted output: %s", path)


def _configure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    sys.exit(main())
