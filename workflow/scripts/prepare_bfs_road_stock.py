"""Derive the road vehicle stock per vehicle category for Switzerland from BFS.

Sources (Swiss Federal Statistical Office, Excel, number of vehicles):
- cc-d-11.03.02.01.03 "Strassenfahrzeugbestand nach Fahrzeuggruppe und Fahrzeugart"
  (BFS/ASTRA, stock at 30 September): one sheet per period with vehicle groups
  (Fahrzeuggruppe) and, below each group, its vehicle types (Fahrzeugart).
- je-d-11.03.02.01.04 "Bestand der Motorfahrräder nach Kanton (inkl. schnelle
  E-Bikes)": mopeds per canton with a national total row (stock usually at
  31 December).

Year headers can carry footnote markers ('2018 2', '2026 p'). Provisional years
('p') are dropped. Rows are matched by normalised group and type label, never by
position; every group total is checked against the sum of its types. Categories
that cannot be derived for a year are left out and logged; nothing is estimated.
"""

import logging
import re

import pandas as pd
from _bfs import HEADER_SEARCH_ROWS, find_year_columns, normalise_label, parse_year
from _schemas import VEHICLE_CATEGORIES, NationalStockSchema

LOGGER = logging.getLogger(__name__)
MIN_YEAR = 1990
PROVISIONAL = "p"
GROUP_HEADER = "Fahrzeuggruppe"
TYPE_HEADER = "Fahrzeugart"

CATEGORY_ROWS = {
    "passenger_car": [("Personenwagen", "")],
    "light_duty": [("Sachentransportfahrzeuge", "Lieferwagen")],
    "heavy_duty": [
        ("Sachentransportfahrzeuge", "Lastwagen"),
        ("Sachentransportfahrzeuge", "Schweres Sattelmotorfahrzeug"),
        ("Sachentransportfahrzeuge", "Sattelschlepper"),
    ],
    "bus": [
        ("Personentransportfahrzeuge", "Gesellschaftswagen"),
        ("Personentransportfahrzeuge", "Kleinbus"),
        ("Personentransportfahrzeuge", "Gelenkbus"),
    ],
    "motorcycle": [("Motorräder", "")],
}
"""Module category -> (group, type) rows summed; type '' is the group total row."""

MOPED_CATEGORY = "motorcycle"
"""Category to which the national moped total is added (both parts required)."""

MOPED_TOTAL = ["Total"]
"""Accepted normalised labels of the national total row of the moped table."""

MOPED_VALUE_HEADERS = ["", "Total"]
"""Accepted sub-headers below a moped year column (other measures are rejected)."""

MOTOR_VEHICLE_TOTAL = [
    "Total Motorfahrzeuge (ohne Motorfahrräder)",
    "Total Motorfahrzeuge",
]
MOTOR_VEHICLE_GROUPS = [
    "Personenwagen",
    "Personentransportfahrzeuge",
    "Sachentransportfahrzeuge",
    "Landwirtschaftsfahrzeuge",
    "Industriefahrzeuge",
    "Motorräder",
]
"""The motor vehicle total row (label aliases across sheets) and its groups."""

CAVEATS = [
    "stock at 30 September; mopeds at the cantonal reference date, usually "
    "31 December (BFS footnote).",
    "from 2018 the stock includes vehicles with a holder address abroad "
    "(15,700 vehicles in 2018; BFS footnote).",
    "2022: about 34,000 vehicles with export plates (26,000 cars) were removed "
    "from the stock database; slight break in comparability (BFS footnote).",
    "bus excludes trolleybuses (BFS footnote); includes minibuses (Kleinbus).",
    "heavy_duty excludes light articulated vehicles (Leichtes "
    "Sattelmotorfahrzeug, at most a few dozen).",
    "motorcycle includes small motorcycles, motor tricycles, quads and "
    "snowmobiles (Motorräder group, about 1,000 snowmobiles) plus mopeds "
    "including fast e-bikes.",
    "mopeds 1980-2013 include federal vehicles (BFS footnote).",
]


def parse_year_header(value) -> tuple[int, str] | None:
    """Return (year, footnote marker) of a header cell, or None if it is no year.

    E.g. 2019.0 -> (2019, ''), '2018 2' -> (2018, '2'), '2026 p' -> (2026, 'p').
    """
    year = parse_year(value)
    if year is not None:
        return year, ""
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d{4})\s+(\S+)\s*", value)
        if match and parse_year(match.group(1)) is not None:
            return int(match.group(1)), match.group(2)
    return None


def find_stock_header(sheet: pd.DataFrame) -> tuple[int, int, int, dict[int, tuple]]:
    """Locate the header of a stock sheet by its column titles.

    Returns the header row, the group and type label columns and a mapping of
    column position -> (year, marker) for the year cells of the header row.
    """
    found = {}
    for row in range(min(HEADER_SEARCH_ROWS, len(sheet))):
        for col in range(sheet.shape[1]):
            label = normalise_label(sheet.iat[row, col])
            if label in (GROUP_HEADER, TYPE_HEADER) and label not in found:
                found[label] = (row, col)
    missing = [h for h in (GROUP_HEADER, TYPE_HEADER) if h not in found]
    if missing:
        raise ValueError(f"No header cells {missing} found.")
    header_row, group_col = found[GROUP_HEADER]
    _, type_col = found[TYPE_HEADER]
    years = {}
    for col in range(sheet.shape[1]):
        parsed = parse_year_header(sheet.iat[header_row, col])
        if parsed is not None:
            years[col] = parsed
    if not years:
        raise ValueError(f"No years in header row {header_row}.")
    if len({year for year, _ in years.values()}) != len(years):
        raise ValueError(f"Duplicated years in header row {header_row}.")
    return header_row, group_col, type_col, years


def tidy_stock_sheet(sheet: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """Turn a raw stock sheet (read with `header=None`) into values per row.

    Returns a frame indexed by (group, type) with one float column per year and
    the footnote marker of each year. Group rows have type ''; type rows belong to
    the last group above them. Rows without a group or type label are skipped.
    """
    header_row, group_col, type_col, years = find_stock_header(sheet)
    keys = []
    rows = []
    group = None
    for row in range(header_row + 1, len(sheet)):
        group_label = normalise_label(sheet.iat[row, group_col])
        type_label = normalise_label(sheet.iat[row, type_col])
        if group_label:
            group = group_label
            key = (group, "")
        elif type_label and group is not None:
            key = (group, type_label)
        else:
            continue
        keys.append(key)
        rows.append([sheet.iat[row, col] for col in years])
    index = pd.MultiIndex.from_tuples(keys, names=["group", "type"])
    duplicated = index[index.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Duplicated rows {duplicated}.")
    columns = [year for year, _ in years.values()]
    values = pd.DataFrame(rows, index=index, columns=columns)
    values = values.apply(pd.to_numeric, errors="coerce").astype(float)
    markers = {year: marker for year, marker in years.values()}
    return values, markers


def read_stock(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Read all sheets of the stock workbook into one frame (rows x years).

    Group totals are checked per sheet. Years before 1990 and provisional years
    are dropped (logged), other footnote markers are logged. Raises if a year
    appears in more than one sheet.
    """
    frames = []
    for name, sheet in sheets.items():
        values, markers = tidy_stock_sheet(sheet)
        check_group_totals(values, name)
        for year, marker in sorted(markers.items()):
            if marker == PROVISIONAL:
                LOGGER.warning(
                    "stock sheet '%s': %s is provisional, dropped.", name, year
                )
            elif marker:
                LOGGER.info("stock sheet '%s': %s has footnote %s.", name, year, marker)
        keep = [
            year
            for year in values.columns
            if markers[year] != PROVISIONAL and year >= MIN_YEAR
        ]
        early = sorted(year for year in values.columns if year < MIN_YEAR)
        if early:
            LOGGER.info("stock sheet '%s': dropped years %s.", name, early)
        frames.append(values[keep])
    years = [year for frame in frames for year in frame.columns]
    duplicated = sorted({year for year in years if years.count(year) > 1})
    if duplicated:
        raise ValueError(f"stock: years {duplicated} appear in several sheets.")
    return pd.concat(frames, axis=1).sort_index(axis=1)


def check_group_totals(stock: pd.DataFrame, sheet_name: str) -> None:
    """Raise where group totals of one sheet differ from the sum of their parts.

    Each group with type rows must equal the sum of its types and the motor
    vehicle total must equal the sum of its groups. Missing type values count as
    zero; years where the total is missing are not checked.
    """
    groups = stock.index.get_level_values("group")
    types = stock.index.get_level_values("type")
    checks = {}
    for group in pd.unique(groups):
        parts = stock[(groups == group) & (types != "")]
        if len(parts):
            checks[group] = (stock.loc[(group, "")], parts.sum(min_count=1))
    totals = [label for label in MOTOR_VEHICLE_TOTAL if (label, "") in stock.index]
    if len(totals) != 1:
        raise KeyError(
            f"stock sheet '{sheet_name}': expected one row labelled "
            f"{MOTOR_VEHICLE_TOTAL}."
        )
    group_rows = [(group, "") for group in MOTOR_VEHICLE_GROUPS]
    missing = [row for row in group_rows if row not in stock.index]
    if missing:
        raise KeyError(f"stock sheet '{sheet_name}': no rows {missing}.")
    checks[totals[0]] = (
        stock.loc[(totals[0], "")],
        stock.loc[group_rows].sum(min_count=1),
    )
    for name, (total, parts) in checks.items():
        off = total.notna() & ((total - parts.fillna(0)).abs() > 0.5)
        if off.any():
            raise ValueError(
                f"stock sheet '{sheet_name}': parts of '{name}' do not sum to "
                f"its total in "
                f"{total.index[off].tolist()}."
            )


def read_mopeds(sheets: dict[str, pd.DataFrame]) -> pd.Series:
    """Read the national moped total per year (from 1990) from the moped workbook.

    Year columns are located by the header; the sub-header below each year must
    be empty or 'Total' (the 'davon schnelle E-Bikes' columns carry no year).
    """
    series = []
    for name, sheet in sheets.items():
        header_row, years = find_year_columns(sheet)
        for col in years:
            sub = normalise_label(sheet.iat[header_row + 1, col])
            if sub not in MOPED_VALUE_HEADERS:
                raise ValueError(
                    f"mopeds sheet '{name}': unexpected sub-header '{sub}' below "
                    f"{years[col]}."
                )
        labels = sheet.iloc[header_row + 1 :, 0].map(normalise_label)
        rows = labels[labels.isin(MOPED_TOTAL)].index
        if len(rows) != 1:
            raise KeyError(
                f"mopeds sheet '{name}': expected one row labelled {MOPED_TOTAL}, "
                f"got {len(rows)}."
            )
        values = pd.to_numeric(sheet.loc[rows[0], list(years)], errors="coerce")
        values.index = [years[col] for col in years]
        series.append(values.astype(float))
    mopeds = pd.concat(series).sort_index()
    duplicated = mopeds.index[mopeds.index.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"mopeds: years {duplicated} appear in several sheets.")
    mopeds = mopeds[mopeds.index >= MIN_YEAR].dropna()
    return mopeds.rename_axis("year")


def derive_categories(stock: pd.DataFrame, mopeds: pd.Series) -> pd.DataFrame:
    """Sum BFS rows into module categories (long `year, vehicle_category, stock`).

    A category is only derived for years where all its rows (and, for
    motorcycles, the moped total) are reported. Raises on a missing row label.
    """
    frames = []
    for category, rows in CATEGORY_ROWS.items():
        missing = [row for row in rows if row not in stock.index]
        if missing:
            raise KeyError(f"{category}: no stock rows {missing}.")
        parts = stock.loc[rows].T
        parts.columns = [f"{group}/{kind}" for group, kind in rows]
        if category == MOPED_CATEGORY:
            parts = parts.join(mopeds.rename("mopeds"), how="outer")
        values = parts.sum(axis=1, min_count=parts.shape[1]).dropna()
        incomplete = parts.index.difference(values.index).tolist()
        if incomplete:
            LOGGER.warning(
                "%s: incomplete BFS values for years %s (left missing).",
                category,
                incomplete,
            )
        frames.append(
            pd.DataFrame(
                {
                    "year": values.index.astype(int),
                    "vehicle_category": category,
                    "stock": values.values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def to_national_stock(
    derived: pd.DataFrame, country_id: str, source: str
) -> pd.DataFrame:
    """Convert derived categories to the validated national stock schema."""
    for category in VEHICLE_CATEGORIES:
        years = derived.loc[derived["vehicle_category"] == category, "year"]
        if years.empty:
            LOGGER.warning("%s %s: no BFS stock (left missing).", country_id, category)
        else:
            LOGGER.info(
                "%s %s: years %s-%s (%d)",
                country_id,
                category,
                years.min(),
                years.max(),
                len(years),
            )
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


def main(stock_path, mopeds_path, country_codes, source, output_path):
    """Read both BFS workbooks, check, derive categories and write parquet."""
    if len(country_codes) != 1:
        raise ValueError(f"BFS covers exactly one country, got {list(country_codes)}.")
    (country_id,) = country_codes
    stock = read_stock(pd.read_excel(stock_path, sheet_name=None, header=None))
    mopeds = read_mopeds(pd.read_excel(mopeds_path, sheet_name=None, header=None))
    for caveat in CAVEATS:
        LOGGER.warning("Caveat: %s", caveat)
    derived = derive_categories(stock, mopeds)
    to_national_stock(derived, country_id, source).to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        stock_path=snakemake.input.stock,  # noqa: F821
        mopeds_path=snakemake.input.mopeds,  # noqa: F821
        country_codes=snakemake.params.country_codes,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        output_path=snakemake.output.stock,  # noqa: F821
    )
