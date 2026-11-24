"""Checksum manifest of the heavy-duty downloads (MULTIYEAR_PLAN.md §4.1, P7).

One row per downloaded file: path, URL, download time (file modification time, UTC),
size and SHA-256, so a run's inputs can be compared with another run's.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from _heavy_duty import sha256

LOGGER = logging.getLogger(__name__)


def manifest(paths: list, urls: list) -> pd.DataFrame:
    """Path, URL, time, size and SHA-256 of each file (`paths` and `urls` aligned)."""
    rows = []
    for path, url in zip(paths, urls, strict=True):
        stat = Path(path).stat()
        rows.append(
            {
                "path": Path(path).as_posix(),
                "url": url,
                "downloaded_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds"),
                "bytes": stat.st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    manifest(list(snakemake.input.files), snakemake.params.urls).to_csv(  # noqa: F821
        snakemake.output.manifest,  # noqa: F821
        index=False,
    )
