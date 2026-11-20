"""Derive the road vehicle stock per vehicle category for Great Britain from DfT.

Source: Department for Transport (DfT) / DVLA table VEH0101 (OpenDocument
spreadsheet), sheet VEH0101a_Lic "Licensed vehicles at the end of the quarter by
body type", in thousands of vehicles. Great Britain from 1994, United Kingdom
from 2014 Q3. Vehicles with a Statutory Off Road Notification (SORN, sheet
VEH0101b) are not part of the stock used here.

The stock of a year is the number of licensed vehicles at the end of Q4
(31 December); years without a Q4 value are dropped. Columns are matched by
header name, never by position, and body types are checked against the reported
totals. Values that are not available ('[x]') are left out and logged.
"""

import logging
import re

import pandas as pd
from _schemas import VEHICLE_CATEGORIES, NationalStockSchema

LOGGER = logging.getLogger(__name__)
MIN_YEAR = 1990
SHEET = "VEH0101a_Lic"
HEADER_SEARCH_ROWS = 15
HEADER = ["Geography", "Date", "Units"]
UNITS = {"Thousands": 1e3}
"""Accepted units -> factor to number of vehicles."""
QUARTER = 4

GEOGRAPHIES = {"GB": "Great Britain", "UK": "United Kingdom"}
"""DfT country code (settings) -> Geography label in VEH0101."""

CATEGORY_COLUMNS = {
    "passenger_car": "Cars",
    "motorcycle": "Motorcycles",
    "light_duty": "Light goods vehicles",
    "heavy_duty": "Heavy goods vehicles",
    "bus": "Buses and coaches",
}
"""Module category -> VEH0101 body type column. 'Other vehicles' is excluded."""

CONSISTENCY_CHECKS = {
    "Total": [
        "Cars",
        "Motorcycles",
        "Light goods vehicles",
        "Heavy goods vehicles",
        "Buses and coaches",
        "Other vehicles",
    ],
    "All goods vehicles": ["Light goods vehicles", "Heavy goods vehicles"],
}
"""Total column -> body type columns that must add up to it."""

CHECK_ATOL = 1.5
"""Tolerance of the sum checks in vehicles (float noise of thousands * 1e3)."""

CAVEATS = [
    "licensed vehicles only: vehicles declared off the road (SORN) are excluded.",
    "stock at 31 December (end of Q4).",
    "bus (Buses and coaches) includes minibuses.",
    "motorcycle includes mopeds, scooters and motor tricycles.",
    "heavy_duty (Heavy goods vehicles, > 3.5 t) includes tractor units.",
    "vehicles are located by the registered keeper's postcode (DfT note).",
]


def find_header_row(sheet: pd.DataFrame) -> int:
    """Return the row whose first cells are the column titles in `HEADER`."""
    for row in range(min(HEADER_SEARCH_ROWS, len(sheet))):
        cells = [str(v).strip() for v in sheet.iloc[row, : len(HEADER)]]
        if cells == HEADER:
            return row
    raise ValueError(f"No header row starting with {HEADER} found.")


def read_licensed(sheet: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw licensed-vehicles sheet (read with `header=None`) into a table.

    Returns columns Geography, Date, Units and one float column per body type;
    values that are not available ('[x]') become NaN. Raises on a missing column.
    """
    row = find_header_row(sheet)
    header = [str(v).strip() if pd.notna(v) else "" for v in sheet.iloc[row]]
    table = sheet.iloc[row + 1 :].copy()
    table.columns = header
    table = table[table["Geography"].notna()]
    needed = set(CATEGORY_COLUMNS.values()) | set(CONSISTENCY_CHECKS)
    needed |= {col for cols in CONSISTENCY_CHECKS.values() for col in cols}
    missing = sorted(needed - set(header))
    if missing:
        raise KeyError(f"VEH0101: no columns {missing}.")
    if len(set(header)) != len(header):
        raise ValueError(f"VEH0101: duplicated columns in {header}.")
    values = table[sorted(needed)].apply(pd.to_numeric, errors="coerce")
    labels = table[HEADER].apply(lambda col: col.astype(str).str.strip())
    return pd.concat([labels, values.astype(float)], axis=1).reset_index(drop=True)


def parse_quarter(date: str) -> tuple[int, int]:
    """Return (year, quarter) of a VEH0101 date such as '2019 Q4 (end December)'."""
    match = re.match(r"\s*(\d{4}) Q([1-4])\b", date)
    if match is None:
        raise ValueError(f"VEH0101: cannot read date '{date}'.")
    return int(match.group(1)), int(match.group(2))


def select_year_end(table: pd.DataFrame, geography: str) -> pd.DataFrame:
    """Keep one geography at the end of Q4 of each year, indexed by year.

    Years without Q4 are dropped (logged). Values are converted to vehicles.
    """
    rows = table[table["Geography"] == geography]
    if rows.empty:
        raise KeyError(f"VEH0101: no rows for geography '{geography}'.")
    quarters = pd.DataFrame(
        rows["Date"].map(parse_quarter).tolist(),
        columns=["year", "quarter"],
        index=rows.index,
    )
    duplicated = quarters.duplicated()
    if duplicated.any():
        raise ValueError(
            f"VEH0101 {geography}: duplicated dates "
            f"{rows.loc[duplicated, 'Date'].tolist()}."
        )
    units = sorted(set(rows["Units"]) - set(UNITS))
    if units:
        raise ValueError(f"VEH0101 {geography}: unexpected units {units}.")
    q4 = quarters["quarter"] == QUARTER
    no_q4 = sorted(set(quarters["year"]) - set(quarters.loc[q4, "year"]))
    if no_q4:
        LOGGER.warning("%s: no Q4 value for years %s, dropped.", geography, no_q4)
    selected = rows[q4].drop(columns=["Geography", "Date"])
    factor = selected.pop("Units").map(UNITS)
    selected = selected.mul(factor, axis=0)
    selected.index = quarters.loc[q4, "year"].values
    return selected.rename_axis("year").sort_index()


def check_totals(stock: pd.DataFrame) -> None:
    """Raise where body types do not add up to the reported totals.

    Years where the total or a body type is missing are not checked.
    """
    for total, parts in CONSISTENCY_CHECKS.items():
        both = pd.concat(
            [stock[total], stock[parts].sum(axis=1, min_count=len(parts))],
            axis=1,
            keys=["total", "sum"],
        ).dropna()
        off = (both["sum"] - both["total"]).abs() > CHECK_ATOL
        if off.any():
            raise ValueError(
                f"VEH0101: {' + '.join(parts)} does not sum to {total} in "
                f"{both.index[off].tolist()}."
            )


def to_national_stock(
    stock: pd.DataFrame, country_id: str, source: str
) -> pd.DataFrame:
    """Map body types to module categories and validate (missing values left out)."""
    stock = stock[stock.index >= MIN_YEAR]
    frames = []
    for category in VEHICLE_CATEGORIES:
        values = stock[CATEGORY_COLUMNS[category]]
        missing = values.index[values.isna()].tolist()
        if missing:
            LOGGER.warning(
                "%s %s: no DfT value for years %s (left missing).",
                country_id,
                category,
                missing,
            )
        values = values.dropna().round()
        if values.empty:
            LOGGER.warning("%s %s: no DfT stock (left missing).", country_id, category)
        else:
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


def main(ods_path, country_codes, source, output_path):
    """Read VEH0101, check, select year-end stock and write parquet."""
    if len(country_codes) != 1:
        raise ValueError(f"DfT covers exactly one country, got {list(country_codes)}.")
    ((country_id, code),) = country_codes.items()
    sheets = pd.read_excel(ods_path, sheet_name=None, header=None, engine="odf")
    if SHEET not in sheets:
        raise KeyError(f"VEH0101: no sheet '{SHEET}' in {list(sheets)}.")
    stock = select_year_end(read_licensed(sheets[SHEET]), GEOGRAPHIES[code])
    check_totals(stock)
    for caveat in CAVEATS:
        LOGGER.warning("Caveat: %s", caveat)
    to_national_stock(stock, country_id, source).to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        ods_path=snakemake.input.ods,  # noqa: F821
        country_codes=snakemake.params.country_codes,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        output_path=snakemake.output.stock,  # noqa: F821
    )
