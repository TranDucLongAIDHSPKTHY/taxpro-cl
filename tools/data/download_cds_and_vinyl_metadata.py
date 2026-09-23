"""Download the three Amazon metadata vintages (2014/2018/2023) for the
CDs_and_Vinyl category, to the exact paths build_cds_and_vinyl_metadata.py
expects (AMAZON_CATEGORY_METADATA_SOURCES[CDS_AND_VINYL_DATASET_NAME]).

Mirrors download.py's download_metadata loop over AMAZON_METADATA_URLS
(the Books-only alias that function is hardwired to), parameterized instead
by AMAZON_CATEGORY_METADATA_URLS[CDS_AND_VINYL_DATASET_NAME] -- the same
extension point build_arts_crafts_and_sewing_metadata.py /
build_musical_instruments_metadata.py rely on. Zero lines of download.py
are modified.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._shared.downloads import configure_logging, download_file, extract_gzip
from config_path.config_path import (
    AMAZON_CATEGORY_METADATA_SOURCES,
    AMAZON_CATEGORY_METADATA_URLS,
    CDS_AND_VINYL_DATASET_NAME,
)


def main():
    logger = configure_logging("INFO")
    urls = AMAZON_CATEGORY_METADATA_URLS[CDS_AND_VINYL_DATASET_NAME]
    sources = AMAZON_CATEGORY_METADATA_SOURCES[CDS_AND_VINYL_DATASET_NAME]
    for year, url in urls.items():
        output = sources[year]
        archive = output.with_name(Path(url).name)
        if output.is_file() and output.stat().st_size > 0:
            logger.info("Skipping metadata that is already extracted: %s", output)
            continue
        download_file(url, archive, force=False, logger=logger)
        extract_gzip(archive, output, force=False, logger=logger)
        if not (output.is_file() and output.stat().st_size > 0):
            raise SystemExit("Extraction failed or produced an empty file: {}".format(output))
        archive.unlink(missing_ok=True)
        logger.info("CDs_and_Vinyl %s metadata ready: %s (%d bytes)", year, output, output.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
