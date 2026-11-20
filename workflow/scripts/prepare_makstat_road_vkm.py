"""Derive heavy-duty road vehicle-km for North Macedonia from MAKSTAT.

Source: MAKSTAT table 560_Trans_MK_T39 "Passenger and freight transport, by years",
row "Kilometres travelled, in '000.000" of the freight block (road freight survey).
The survey covers goods vehicles registered in North Macedonia with >= 3.5 t load
capacity, in and outside the country (nationality principle); transit and cabotage
are included from 2018. These are taken as heavy_duty. Eurostat publishes the same
values as "lorries and road trains".

Years where the total is not the sum of hire-or-reward and own-account km
(2000-2003: no split, much smaller totals) are dropped and logged.
"""

import logging

import pandas as pd
from _pxweb import log_updated, read_px_csv
from _schemas import NationalVkmSchema

LOGGER = logging.getLogger(__name__)
MKM_TO_KM = 1e6
TOTAL = "Kilometres travelled, in '000.000"
PARTS = [
    "Of that, hire or reward, in '000.000 km",
    "Of that, on own account, in '000.000 km",
]
CHECK_ATOL = 0.5
"""Tolerance of the split check in million km (values are rounded to integers)."""


def heavy_vehicle_km(table: pd.DataFrame) -> pd.Series:
    """Freight vehicle-km in million km per year, for years with a consistent split."""
    missing = sorted({TOTAL, *PARTS} - set(table.index))
    if missing:
        raise KeyError(f"MAKSTAT freight table: no rows {missing}.")
    total = table.loc[TOTAL]
    parts = table.loc[PARTS].sum(min_count=len(PARTS))
    consistent = (total - parts).abs() <= CHECK_ATOL
    dropped = total.index[~consistent & total.notna()].tolist()
    if dropped:
        LOGGER.warning(
            "MKD heavy_duty: total km is not hire + own account in %s, dropped.",
            dropped,
        )
    kept = total[consistent]
    kept.index = kept.index.astype(int)
    return kept.rename_axis("year").rename("mvkm")


def to_national_vkm(
    mvkm: pd.Series, country_id: str, principle: str, source: str
) -> pd.DataFrame:
    """Convert to the national vehicle-km schema."""
    if mvkm.empty:
        raise ValueError("No MAKSTAT freight vehicle-km left.")
    LOGGER.info(
        "%s heavy_duty: years %s-%s (%d)",
        country_id,
        mvkm.index.min(),
        mvkm.index.max(),
        len(mvkm),
    )
    national = pd.DataFrame(
        {
            "country_id": country_id,
            "year": mvkm.index.astype(int),
            "vehicle_category": "heavy_duty",
            "vkm": mvkm.to_numpy(dtype=float) * MKM_TO_KM,
            "unit": "vkm",
            "source": source,
            "principle": principle,
        }
    )
    return NationalVkmSchema.validate(national)


def main(freight_path, listing_path, table_id, country_codes, principles, source, output_path):
    """Read the freight table, select heavy-duty vehicle-km and write parquet."""
    if len(country_codes) != 1:
        raise ValueError(f"MAKSTAT covers exactly one country, got {list(country_codes)}.")
    (country_id,) = country_codes
    log_updated(listing_path, table_id)
    mvkm = heavy_vehicle_km(read_px_csv(freight_path))
    national = to_national_vkm(mvkm, country_id, principles[country_id], source)
    national.to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        freight_path=snakemake.input.freight,  # noqa: F821
        listing_path=snakemake.input.listing,  # noqa: F821
        table_id=snakemake.params.table_id,  # noqa: F821
        country_codes=snakemake.params.country_codes,  # noqa: F821
        principles=snakemake.params.principles,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        output_path=snakemake.output.vkm,  # noqa: F821
    )
