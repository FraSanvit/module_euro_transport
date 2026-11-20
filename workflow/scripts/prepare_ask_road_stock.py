"""Derive the road vehicle stock per vehicle category for Kosovo from ASK.

Source: Kosovo Agency of Statistics (ASK) table tr05 "Registered motor and non motor
vehicles", one row per vehicle type and one column per year, plus a total row that
is checked against the sum of all types.

Categories:
- passenger_car = Cars;
- heavy_duty = Transport vehicles >= 3.5 t (includes road tractors; equals Eurostat
  lorries > 3.5 t + road tractors from 2022);
- light_duty = Transport vehicles < 3.5 t;
- bus = Buses + Van ("Kombibuset", minibuses).
Motorcycles are not derived: no mopeds are reported. "Tractors" (not road tractors)
and trailers are not used. Unreliable years listed in settings (`exclude`) are
dropped and logged.
"""

import logging

import pandas as pd
from _national import drop_excluded
from _pxweb import log_updated, read_px_csv
from _schemas import VEHICLE_CATEGORIES, NationalStockSchema

LOGGER = logging.getLogger(__name__)
TOTAL = "Total"
TYPES = [
    "Cars",
    "Transport veh., 3.5 and over 3.5t",
    "Transport veh. under 3.5t",
    "Van",
    "Buses",
    "Motorcycles",
    "Tractors",
    "Trailer under 3.5t",
    "Trailer 3.5 and above 3.5t",
]
CATEGORY_ROWS = {
    "passenger_car": ["Cars"],
    "heavy_duty": ["Transport veh., 3.5 and over 3.5t"],
    "light_duty": ["Transport veh. under 3.5t"],
    "bus": ["Buses", "Van"],
}
"""Module category -> tr05 rows that add up to it."""
NOT_DERIVED = {"motorcycle": "no mopeds reported"}
CHECK_ATOL = 0.5


def check_total(table: pd.DataFrame) -> None:
    """Raise where the vehicle types do not add up to the total row."""
    missing = sorted({TOTAL, *TYPES} - set(table.index))
    if missing:
        raise KeyError(f"tr05: no rows {missing}.")
    both = pd.concat(
        [table.loc[TOTAL], table.loc[TYPES].sum(min_count=len(TYPES))],
        axis=1,
        keys=["total", "sum"],
    ).dropna()
    off = (both["total"] - both["sum"]).abs() > CHECK_ATOL
    if off.any():
        raise ValueError(f"tr05: types do not sum to the total in {both.index[off].tolist()}.")


def derive_categories(table: pd.DataFrame, country_id: str) -> pd.DataFrame:
    """Sum the rows per category (long `year, vehicle_category, stock`)."""
    frames = []
    for category in VEHICLE_CATEGORIES:
        if category in NOT_DERIVED:
            LOGGER.warning(
                "%s %s: not derived, %s (left missing).",
                country_id,
                category,
                NOT_DERIVED[category],
            )
            continue
        rows = CATEGORY_ROWS[category]
        values = table.loc[rows].sum(min_count=len(rows)).dropna()
        values.index = values.index.astype(int)
        LOGGER.info(
            "%s %s: years %s-%s (%d)",
            country_id,
            category,
            values.index.min(),
            values.index.max(),
            len(values),
        )
        frames.append(
            pd.DataFrame(
                {"year": values.index, "vehicle_category": category, "stock": values}
            )
        )
    return pd.concat(frames, ignore_index=True)


def to_national_stock(
    derived: pd.DataFrame, country_id: str, source: str
) -> pd.DataFrame:
    """Convert to the national stock schema and validate."""
    national = pd.DataFrame(
        {
            "country_id": country_id,
            "year": derived["year"].astype(int),
            "vehicle_category": derived["vehicle_category"],
            "stock": derived["stock"].astype(float),
            "unit": "vehicles",
            "source": source,
        }
    ).sort_values(["country_id", "year", "vehicle_category"], ignore_index=True)
    return NationalStockSchema.validate(national)


def main(stock_path, listing_path, table_id, country_codes, source, exclude, output_path):
    """Read tr05, check, derive categories, drop unreliable years and write parquet."""
    if len(country_codes) != 1:
        raise ValueError(f"tr05 covers one country, got {list(country_codes)}.")
    (country_id,) = country_codes
    log_updated(listing_path, table_id)
    table = read_px_csv(stock_path)
    check_total(table)
    stock = to_national_stock(derive_categories(table, country_id), country_id, source)
    drop_excluded(stock, exclude).to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        stock_path=snakemake.input.stock,  # noqa: F821
        listing_path=snakemake.input.listing,  # noqa: F821
        table_id=snakemake.params.table_id,  # noqa: F821
        country_codes=snakemake.params.country_codes,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        exclude=snakemake.params.exclude,  # noqa: F821
        output_path=snakemake.output.stock,  # noqa: F821
    )
