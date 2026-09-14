"""Helpers for downloading and extracting project data."""

from __future__ import annotations

import gzip
import logging
import sys
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# Allow direct imports when a tool module is launched outside module mode.
_PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_IMPORT_ROOT))

from config_path.config_path import temporary_file


CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 128 * 1024 * 1024


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    logger_name: str = "preprocessing",
) -> logging.Logger:
    """Configure preprocessing logs for both console and UTF-8 files."""
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers = [console_handler]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(logger_name)


def download_file(
    url: str,
    destination: Path,
    *,
    force: bool,
    logger: logging.Logger,
    retries: int = 3,
    timeout: int = 120,
) -> bool:
    """Atomically download a URL; return True when a download occurred."""
    destination = Path(destination)
    if destination.is_file() and not force:
        logger.info("Skipping existing file: %s", destination)
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(destination, "download")

    for attempt in range(1, retries + 1):
        try:
            if temporary.exists():
                temporary.unlink()
            logger.info("Download attempt %d/%d: %s", attempt, retries, url)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ID-GRec-data-downloader/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
                total = _content_length(response.headers.get("Content-Length"))
                downloaded = 0
                next_report = PROGRESS_INTERVAL
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        _log_progress(logger, destination, downloaded, total)
                        next_report += PROGRESS_INTERVAL

            if total is not None and downloaded != total:
                raise IOError(
                    "Downloaded size mismatch for {}: received {}, expected {}".format(
                        destination, downloaded, total
                    )
                )
            os.replace(str(temporary), str(destination))
            _log_progress(logger, destination, downloaded, total, completed=True)
            return True
        except Exception as error:
            if temporary.exists():
                temporary.unlink()
            if attempt == retries:
                logger.error("Download failed after %d attempts: %s", retries, destination)
                raise
            delay = 2 ** (attempt - 1)
            logger.warning("Download error for %s: %s; retrying in %d seconds", destination, error, delay)
            time.sleep(delay)

    return False


def extract_gzip(
    archive: Path,
    destination: Path,
    *,
    force: bool,
    logger: logging.Logger,
) -> bool:
    """Atomically extract one gzip file."""
    archive, destination = Path(archive), Path(destination)
    if destination.is_file() and not force:
        logger.info("Skipping extracted file: %s", destination)
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(destination, "download")
    if temporary.exists():
        temporary.unlink()

    logger.info("Extracting gzip: %s -> %s", archive, destination)
    try:
        with gzip.open(str(archive), "rb") as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=CHUNK_SIZE)
        os.replace(str(temporary), str(destination))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    logger.info("Extraction completed: %s", destination)
    return True


def extract_zip(
    archive: Path,
    destination: Path,
    *,
    force: bool,
    logger: logging.Logger,
) -> bool:
    """Extract ZIP contents while preserving paths and blocking traversal."""
    archive, destination = Path(archive), Path(destination)
    if _directory_has_files(destination) and not force:
        logger.info("Skipping extracted directory: %s", destination)
        return False

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    logger.info("Extracting ZIP: %s -> %s", archive, destination)
    with zipfile.ZipFile(str(archive)) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError("Unsafe path in ZIP archive: {}".format(member.filename))
        zip_file.extractall(str(destination))
    logger.info("Extraction completed: %s", destination)
    return True


def _directory_has_files(path: Path) -> bool:
    return Path(path).is_dir() and any(item.is_file() for item in Path(path).rglob("*"))


def _content_length(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _log_progress(
    logger: logging.Logger,
    destination: Path,
    downloaded: int,
    total: Optional[int],
    completed: bool = False,
) -> None:
    downloaded_mb = downloaded / (1024 * 1024)
    prefix = "Download completed" if completed else "Download progress"
    if total:
        logger.info(
            "%s %s: %.1f/%.1f MiB (%.1f%%)",
            prefix,
            destination,
            downloaded_mb,
            total / (1024 * 1024),
            downloaded * 100.0 / total,
        )
    else:
        logger.info("%s %s: %.1f MiB", prefix, destination, downloaded_mb)
