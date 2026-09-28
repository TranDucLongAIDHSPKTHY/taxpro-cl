"""Stdlib-only (no pandas) streaming CSV upsert helper.

The project's runtime environment (see requirements.txt) does not include
pandas, and ranking_details.csv can grow into the millions of rows for a
full-population sweep -- so both the "read what's already there" and the
"write the merged result" sides are done row-by-row with csv.DictReader /
csv.DictWriter rather than loading a whole table into memory.
"""

from __future__ import annotations

import csv
from pathlib import Path


def upsert_csv(path, fieldnames, new_rows, match_predicate):
    """Merge new_rows into the CSV at path, replacing any existing row that
    match_predicate accepts.

    new_rows may be a small in-memory list (summary/manifest tables) or a
    generator streaming from a temporary file (the details table) -- either
    way this function only ever holds one row at a time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".merge_tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as out_stream:
        writer = csv.DictWriter(out_stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as old_stream:
                for row in csv.DictReader(old_stream):
                    if not match_predicate(row):
                        writer.writerow({key: row.get(key, "") for key in fieldnames})
        for row in new_rows:
            writer.writerow(row)
    tmp_path.replace(path)


def stream_rows(path):
    """Yield rows (as dicts) from an existing CSV file, one at a time."""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)
