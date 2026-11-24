"""German motorway toll km by vehicle nationality from the project's Zenodo record (MULTIYEAR_PLAN.md §4.1).

The BALM toll statistics are published only as PDF; the parsed series is archived on
Zenodo. The zip is checked against its md5 and the CSV against its SHA-256. Returns
`country_id, year, toll_domestic, toll_foreign` [vkm] (motorways, BAB).
"""

import hashlib
import io
import logging
import zipfile

import pandas as pd

LOGGER = logging.getLogger(__name__)
THS_KM = 1e3


def md5(path, chunk=1 << 24) -> str:
    """md5 hex digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_toll(zip_path, member: str, expected_sha256: str) -> pd.DataFrame:
    """Extract and check the toll CSV from the record zip."""
    with zipfile.ZipFile(zip_path) as archive:
        content = archive.read(member)
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{member}: SHA-256 {actual} differs from {expected_sha256}.")
    table = pd.read_csv(io.BytesIO(content))
    LOGGER.info("DE toll series: years %d-%d", table["year"].min(), table["year"].max())
    return table


def main(zip_path, expected_md5, member, expected_sha256, output_path):
    """Check the record, extract the series and write parquet (vkm)."""
    actual = md5(zip_path)
    if actual != expected_md5:
        raise ValueError(f"{zip_path}: md5 {actual} differs from {expected_md5}.")
    table = read_toll(zip_path, member, expected_sha256)
    out = pd.DataFrame(
        {
            "country_id": "DEU",
            "year": table["year"].astype(int),
            "toll_domestic": table["bab_domestic_1000km"] * THS_KM,
            "toll_foreign": table["bab_foreign_1000km"] * THS_KM,
        }
    )
    out.to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        zip_path=snakemake.input.record,  # noqa: F821
        expected_md5=snakemake.params.md5,  # noqa: F821
        member=snakemake.params.member,  # noqa: F821
        expected_sha256=snakemake.params.sha256,  # noqa: F821
        output_path=snakemake.output.toll,  # noqa: F821
    )
