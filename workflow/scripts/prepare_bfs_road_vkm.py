"""Derive annual road vehicle-km per vehicle category for Switzerland from BFS.

Sources (Swiss Federal Statistical Office, Excel, million vehicle-km):
- je-d-11.05.01.01 "Fahrleistungen im Güterverkehr" (freight): light (<= 3.5 t)
  and heavy (> 3.5 t) goods vehicles, domestic and foreign vehicles in Switzerland.
- je-d-11.04.01.01 "Fahrleistungen und Fahrzeugbewegungen im Personenverkehr"
  (passenger): private road traffic by vehicle type and public buses/trolleybuses.

Each workbook has one sheet per period. Year headers are interleaved with
footnote-flag columns, missing values are strings such as '…' and row labels carry
footnote markers, so rows are matched by normalised label, never by position.
Years before 1990 are outside the module scope and are dropped. Categories that
cannot be derived for a year are left out and logged; nothing is estimated.
"""

import logging

import pandas as pd
from _bfs import find_year_columns, normalise_label
from _schemas import VEHICLE_CATEGORIES, NationalVkmSchema

LOGGER = logging.getLogger(__name__)
MKM_TO_KM = 1e6
MIN_YEAR = 1990
CHECK_RTOL = 1e-3

FREIGHT_ROWS = {
    "total": ["Total in- und ausländische Fahrzeuge"],
    "light": ["Leichte Fahrzeuge"],
    "heavy": ["Schwere Fahrzeuge"],
}
"""Freight components -> accepted normalised row labels (aliases across sheets)."""

PASSENGER_ROWS = {
    "private_total": ["Total"],
    "car": ["Personenwagen"],
    "coach": ["Privatcars"],
    "motorcycle": ["Motorräder"],
    "moped": ["Motorfahrräder, inkl. schnelle E-Bikes", "Motorfahrräder"],
    "bus": ["Autobusse", "Busverkehr"],
    "trolleybus": ["Trolleybusse"],
}
"""Passenger components -> accepted normalised row labels (aliases across sheets)."""

CATEGORY_COMPONENTS = {
    "heavy_duty": ("freight", ["heavy"]),
    "light_duty": ("freight", ["light"]),
    "passenger_car": ("passenger", ["car"]),
    "motorcycle": ("passenger", ["motorcycle", "moped"]),
    "bus": ("passenger", ["bus", "trolleybus", "coach"]),
}
"""Module category -> (table, components summed). Trams are excluded."""

CONSISTENCY_CHECKS = {
    "freight": ("total", ["light", "heavy"]),
    "passenger": ("private_total", ["car", "coach", "motorcycle", "moped"]),
}
"""Table -> (total component, components that must add up to it)."""

CAVEATS = [
    "light_duty excludes foreign light goods vehicles (BFS freight footnote).",
    "bus: public buses and trolleybuses are reported in Kurs-km (course-km of "
    "the vehicles, 1981-2007 including trailers); private coaches in vehicle-km.",
    "passenger private traffic excludes minibuses and motorhomes (BFS footnote).",
]


def tidy_sheet(sheet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a raw sheet (read with `header=None`) into values and flags.

    Returns two frames indexed by normalised row label with one column per year:
    numeric values in million (missing strings such as '…' become NaN) and the
    footnote flag of each value (from the flag column right after a year, '' if none).
    """
    header_row, years = find_year_columns(sheet)
    body = sheet.iloc[header_row + 1 :]
    labels = body.iloc[:, 0].map(normalise_label)
    body = body[labels != ""]
    labels = labels[labels != ""]
    values = {}
    flags = {}
    for col, year in years.items():
        values[year] = pd.to_numeric(body.iloc[:, col], errors="coerce")
        flag_col = col + 1
        if flag_col < sheet.shape[1] and flag_col not in years:
            header = sheet.iat[header_row, flag_col]
            if pd.isna(header):
                flag = body.iloc[:, flag_col]
                flags[year] = flag.where(flag.notna(), "").astype(str).str.strip()
                continue
        flags[year] = pd.Series("", index=body.index)
    index = pd.Index(labels, name="label")
    values = pd.DataFrame(values).set_index(index).astype(float)
    flags = pd.DataFrame(flags).set_index(index)
    return values, flags


def select_rows(
    values: pd.DataFrame, rows: dict[str, list[str]], sheet_name: str
) -> pd.DataFrame:
    """Pick one row per component by its accepted labels.

    Returns a frame indexed by year with one column per component. Raises if a
    component matches no row or more than one row.
    """
    selected = {}
    for component, aliases in rows.items():
        matches = values[values.index.isin(aliases)]
        if len(matches) == 0:
            raise KeyError(
                f"Sheet '{sheet_name}': no row labelled {aliases} for '{component}'."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Sheet '{sheet_name}': {len(matches)} rows labelled {aliases} "
                f"for '{component}'."
            )
        selected[component] = matches.iloc[0]
    table = pd.DataFrame(selected)
    table.index = table.index.astype(int)
    return table.rename_axis(index="year")


def read_table(
    sheets: dict[str, pd.DataFrame], rows: dict[str, list[str]], table: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the required components from all sheets of one BFS workbook.

    Sheets that only cover years before 1990 are skipped (logged); earlier years
    in other sheets are dropped. Returns values (Mvkm) and flags, indexed by year.
    Raises if a year appears in more than one sheet.
    """
    values = []
    flags = []
    for name, sheet in sheets.items():
        sheet_values, sheet_flags = tidy_sheet(sheet)
        years = sorted(sheet_values.columns)
        if max(years) < MIN_YEAR:
            LOGGER.info(
                "%s sheet '%s': skipped, years %s-%s are before %s.",
                table,
                name,
                min(years),
                max(years),
                MIN_YEAR,
            )
            continue
        early = [year for year in years if year < MIN_YEAR]
        if early:
            LOGGER.info("%s sheet '%s': dropped years %s.", table, name, early)
        keep = [year for year in years if year >= MIN_YEAR]
        values.append(select_rows(sheet_values[keep], rows, name))
        flags.append(select_rows(sheet_flags[keep], rows, name))
    if not values:
        raise ValueError(f"{table}: no sheet with years from {MIN_YEAR}.")
    values = pd.concat(values).sort_index()
    flags = pd.concat(flags).sort_index()
    duplicated = values.index[values.index.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{table}: years {duplicated} appear in several sheets.")
    return values, flags


def check_totals(
    values: pd.DataFrame, total: str, parts: list[str], table: str
) -> None:
    """Raise where the components do not add up to the reported total.

    Years where the total or a component is missing are not checked.
    """
    both = pd.concat(
        [values[total], values[parts].sum(axis=1, min_count=len(parts))],
        axis=1,
        keys=["total", "sum"],
    ).dropna()
    off = ~((both["sum"] - both["total"]).abs() <= CHECK_RTOL * both["total"].abs())
    if off.any():
        raise ValueError(
            f"{table}: {' + '.join(parts)} does not sum to {total} "
            f"(rtol {CHECK_RTOL}) in {both.index[off].tolist()}."
        )


def log_flags(flags: pd.DataFrame, table: str) -> None:
    """Log how many values of each component carry each footnote flag."""
    for component in flags.columns:
        counts = flags[component][flags[component] != ""].value_counts()
        for flag, count in counts.sort_index().items():
            LOGGER.info("%s %s: %d values flagged '%s'", table, component, count, flag)


def derive_categories(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sum BFS components into module categories.

    A category is only derived for years where all its components are reported.
    Returns long columns `year, vehicle_category, mvkm`.
    """
    frames = []
    for category, (table, components) in CATEGORY_COMPONENTS.items():
        values = tables[table][components]
        mvkm = values.sum(axis=1, min_count=len(components)).dropna()
        missing = values.index.difference(mvkm.index).tolist()
        if missing:
            LOGGER.warning(
                "%s: no BFS value for years %s (left missing).", category, missing
            )
        frames.append(
            pd.DataFrame(
                {"year": mvkm.index, "vehicle_category": category, "mvkm": mvkm.values}
            )
        )
    return pd.concat(frames, ignore_index=True)


def to_national_vkm(
    derived: pd.DataFrame, country_id: str, principle: str, source: str
) -> pd.DataFrame:
    """Convert derived categories to the validated national vehicle-km schema."""
    for category, group in derived.groupby("vehicle_category"):
        LOGGER.info(
            "%s %s: years %s-%s (%d)",
            country_id,
            category,
            group["year"].min(),
            group["year"].max(),
            len(group),
        )
    missing = sorted(set(VEHICLE_CATEGORIES) - set(derived["vehicle_category"]))
    if missing:
        LOGGER.warning("%s: no BFS data for %s (left missing).", country_id, missing)
    national = pd.DataFrame(
        {
            "country_id": country_id,
            "year": derived["year"].astype(int),
            "vehicle_category": derived["vehicle_category"],
            "vkm": derived["mvkm"].astype(float) * MKM_TO_KM,
            "unit": "vkm",
            "source": source,
            "principle": principle,
        }
    )
    national = national.sort_values(
        ["country_id", "year", "vehicle_category"], ignore_index=True
    )
    return NationalVkmSchema.validate(national)


def main(freight_path, passenger_path, country_codes, principles, source, output_path):
    """Read both BFS workbooks, derive categories and write parquet."""
    if len(country_codes) != 1:
        raise ValueError(f"BFS covers exactly one country, got {list(country_codes)}.")
    (country_id,) = country_codes
    tables = {}
    for table, path, rows in [
        ("freight", freight_path, FREIGHT_ROWS),
        ("passenger", passenger_path, PASSENGER_ROWS),
    ]:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
        values, flags = read_table(sheets, rows, table)
        total, parts = CONSISTENCY_CHECKS[table]
        check_totals(values, total, parts, table)
        log_flags(flags, table)
        tables[table] = values
    for caveat in CAVEATS:
        LOGGER.warning("Caveat: %s", caveat)
    derived = derive_categories(tables)
    national = to_national_vkm(derived, country_id, principles[country_id], source)
    national.to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        freight_path=snakemake.input.freight,  # noqa: F821
        passenger_path=snakemake.input.passenger,  # noqa: F821
        country_codes=snakemake.params.country_codes,  # noqa: F821
        principles=snakemake.params.principles,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        output_path=snakemake.output.vkm,  # noqa: F821
    )
