"""Derive the road vehicle stock per vehicle category for North Macedonia from MAKSTAT.

Source: MAKSTAT table 150_Trans_regi_veh_by_type_mun "Registered road motor
vehicles and trailers by years, by municipalities, number" (stock at 31 December,
2012 onwards). Columns are "<year> <type>"; the national total is one row.

Only passenger cars and buses map to module categories. Goods vehicles are not
split by weight (heavy_duty and light_duty cannot be derived), and motor-cycles
exclude mopeds (motorcycle includes mopeds in this module): these are left missing
and logged. The total is checked against the sum of all vehicle types.
"""

import logging
import re

import pandas as pd
from _makstat import log_updated, read_px_csv
from _schemas import VEHICLE_CATEGORIES, NationalStockSchema

LOGGER = logging.getLogger(__name__)
TOTAL = "Total"
TYPES = [
    "motor-cycles",
    "passenger cars",
    "buses",
    "goods vehicles",
    "road tractors",
    "tractors",
    "work vehicles",
    "trailers",
]
CATEGORY_TYPES = {"passenger_car": "passenger cars", "bus": "buses"}
"""Module category -> MAKSTAT vehicle type."""
NOT_DERIVED = {
    "heavy_duty": "goods vehicles are not split by weight",
    "light_duty": "goods vehicles are not split by weight",
    "motorcycle": "motor-cycles exclude mopeds",
}
CHECK_ATOL = 0.5
"""Tolerance of the total check in vehicles."""


def national_by_year(table: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return the national row as a year x vehicle type table."""
    if label not in table.index:
        raise KeyError(f"MAKSTAT stock table: no row '{label}'.")
    row = table.loc[label]
    parsed = [re.fullmatch(r"(\d{4}) (.+)", str(col)) for col in row.index]
    bad = [col for col, match in zip(row.index, parsed) if match is None]
    if bad:
        raise ValueError(f"MAKSTAT stock table: cannot read columns {bad}.")
    index = pd.MultiIndex.from_tuples(
        [(int(m.group(1)), m.group(2).strip()) for m in parsed], names=["year", "type"]
    )
    wide = pd.Series(row.to_numpy(dtype=float), index=index).unstack("type")
    missing = sorted({TOTAL, *TYPES} - set(wide.columns))
    if missing:
        raise KeyError(f"MAKSTAT stock table: no vehicle types {missing}.")
    return wide


def check_total(wide: pd.DataFrame) -> None:
    """Raise where the vehicle types do not add up to the total."""
    both = pd.concat(
        [wide[TOTAL], wide[TYPES].sum(axis=1, min_count=len(TYPES))],
        axis=1,
        keys=["total", "sum"],
    ).dropna()
    off = (both["total"] - both["sum"]).abs() > CHECK_ATOL
    if off.any():
        raise ValueError(
            f"MAKSTAT stock: vehicle types do not sum to the total in "
            f"{both.index[off].tolist()}."
        )


def to_national_stock(wide: pd.DataFrame, country_id: str, source: str) -> pd.DataFrame:
    """Map vehicle types to module categories and validate (missing values left out)."""
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
        values = wide[CATEGORY_TYPES[category]].dropna()
        if values.empty:
            LOGGER.warning("%s %s: no MAKSTAT stock (left missing).", country_id, category)
            continue
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
    derived = pd.concat(frames, ignore_index=True)
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
    """Read the registered vehicles table, check, map and write parquet."""
    if len(country_codes) != 1:
        raise ValueError(f"MAKSTAT covers exactly one country, got {list(country_codes)}.")
    ((country_id, label),) = country_codes.items()
    log_updated(listing_path, table_id)
    wide = national_by_year(read_px_csv(stock_path), label)
    check_total(wide)
    to_national_stock(wide, country_id, source).to_parquet(output_path, index=False)


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
