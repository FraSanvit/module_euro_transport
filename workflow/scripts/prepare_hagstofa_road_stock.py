"""Derive the road vehicle stock per vehicle category for Iceland from Statistics Iceland.

Source: Statistics Iceland (Hagstofa) table SAM03105 "Registered motor vehicles"
(vehicles in use at 31 December, from the Icelandic Transport Authority register),
one row per year and one column per EU vehicle class. Vehicles temporarily off the
register are excluded. Eurostat's Icelandic motorcycle and moped counts use an older
"all registered" basis and are not used.

Categories (all parts required):
- passenger_car = M1; bus = M2 + M3; light_duty = N1;
- heavy_duty = N2 + N3 (road tractors are N3);
- motorcycle = L1e (mopeds) + L2 + L3e + L3e with sidecar + L5e.
"""

import logging

import pandas as pd
from _pxweb import log_updated, read_px_csv
from _schemas import VEHICLE_CATEGORIES, NationalStockSchema

LOGGER = logging.getLogger(__name__)
CATEGORY_COLUMNS = {
    "passenger_car": ["Passenger cars, 1-8 passengers (M1)"],
    "bus": ["Buses <5t. total weight (M2)", "Buses >5t. total weight (M3)"],
    "light_duty": ["Vans (N1)"],
    "heavy_duty": ["Lorries I <12t. (N2)", "Lorries II >12t. (N3)"],
    "motorcycle": [
        "Light motorbikes I (L1e)",
        "Light motorbikes II (L2)",
        "Heavy motorbikes I (L3e)",
        "Heavy motorbikes II with sidecar (L3e)",
        "Heavy motorbikes III on three wheels (L5e)",
    ],
}
"""Module category -> SAM03105 columns that add up to it."""


def derive_categories(table: pd.DataFrame) -> pd.DataFrame:
    """Sum the vehicle classes per category (long `year, vehicle_category, stock`)."""
    needed = [col for cols in CATEGORY_COLUMNS.values() for col in cols]
    missing = sorted(set(needed) - set(table.columns))
    if missing:
        raise KeyError(f"SAM03105: no columns {missing}.")
    years = pd.Index(table.index.astype(int), name="year")
    frames = []
    for category in VEHICLE_CATEGORIES:
        columns = CATEGORY_COLUMNS[category]
        values = table[columns].sum(axis=1, min_count=len(columns))
        values.index = years
        values = values.dropna()
        if values.empty:
            LOGGER.warning("ISL %s: no SAM03105 stock (left missing).", category)
            continue
        LOGGER.info(
            "ISL %s: years %s-%s (%d)",
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


def main(stock_path, listing_path, table_id, country_codes, source, output_path):
    """Read SAM03105, derive categories and write parquet."""
    if len(country_codes) != 1:
        raise ValueError(f"SAM03105 covers one country, got {list(country_codes)}.")
    (country_id,) = country_codes
    log_updated(listing_path, table_id)
    derived = derive_categories(read_px_csv(stock_path))
    to_national_stock(derived, country_id, source).to_parquet(output_path, index=False)


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
        output_path=snakemake.output.stock,  # noqa: F821
    )
