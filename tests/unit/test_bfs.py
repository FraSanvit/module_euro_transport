"""Unit tests for the BFS (Switzerland) road vehicle-km import."""

import logging

import numpy as np
import pandas as pd
import pandera.errors
import pytest
from _bfs import find_year_columns, normalise_label, parse_year
from prepare_bfs_road_vkm import (
    FREIGHT_ROWS,
    PASSENGER_ROWS,
    check_totals,
    derive_categories,
    main,
    read_table,
    select_rows,
    tidy_sheet,
    to_national_vkm,
)

MISSING = "…"


def freight_sheet(years, light, heavy, flag_years=(), header_row=1, label_suffix="3"):
    """Raw freight sheet as pandas reads it with `header=None`.

    Years in `flag_years` get a footnote-flag column right after them.
    """
    header = [None]
    columns = []
    for year in years:
        header.append(year)
        columns.append(year)
        if year in flag_years:
            header.append(None)
            columns.append(("flag", year))

    def row(label, values, flag=None):
        cells = [label]
        for col in columns:
            if isinstance(col, tuple):
                cells.append(flag)
            else:
                cells.append(values[years.index(col)])
        return cells

    def total(i):
        if isinstance(light[i], str) or isinstance(heavy[i], str):
            return MISSING
        return light[i] + heavy[i]

    rows = [["Fahrleistungen im Güterverkehr"] + [None] * len(columns)]
    rows += [[None] * (len(columns) + 1)] * (header_row - 1)
    rows += [header]
    rows += [
        ["Schiene1, in Millionen Zugskilometern"] + [None] * len(columns),
        row("Total Güterzüge", [30.0] * len(years), flag="5"),
        ["Strasse2, in Millionen Fahrzeugkilometern"] + [None] * len(columns),
        row(
            "Total in- und ausländische Fahrzeuge",
            [total(i) for i in range(len(years))],
        ),
        row(f"Leichte Fahrzeuge{label_suffix}", light),
        row("Schwere Fahrzeuge 4)", heavy, flag="r"),
        row("davon Lastwagen", [1.0] * len(years)),
        ["3 Gesamtgewicht <=3,5t"] + [None] * len(columns),
        ["Stand: Juli 2026"] + [None] * len(columns),
    ]
    return pd.DataFrame(rows)


def passenger_sheet(
    years, values, moped_label="   Motorfahrräder, inkl. schnelle E-Bikes 5)"
):
    """Raw passenger sheet with a flag column after every year.

    `values` maps component -> list of values per year.
    """
    header = [None]
    for year in years:
        header += [year, None]
    width = len(header)

    def row(label, component, flag=None):
        cells = [label]
        for i in range(len(years)):
            cells += [values[component][i], flag]
        return cells

    def total(i):
        parts = [values[c][i] for c in ["car", "coach", "motorcycle", "moped"]]
        return MISSING if any(isinstance(p, str) for p in parts) else sum(parts)

    values = {**values, "private_total": [total(i) for i in range(len(years))]}
    empty = [None] * (width - 1)
    rows = [
        ["Fahrleistungen und Fahrzeugbewegungen im Personenverkehr"] + empty,
        header,
        ["Strasse"] + empty,
        ["Öffentlicher Verkehr"] + empty,
        ["Total in Mio. Kurs-km"] + [999.0, None] * len(years),
        ["Tram 2) "] + [50.0, None] * len(years),
        row("Trolleybusse 2)", "trolleybus"),
        row("Autobusse 3)", "bus"),
        ["Privater Verkehr 4) in Mio. Fahrzeug-km"] + empty,
        row("Total ", "private_total", flag="10)"),
        row("   Personenwagen", "car", flag="10)"),
        ["  inländische Personenwagen"] + [1.0, None] * len(years),
        ["  ausländische Personenwagen"] + [2.0, None] * len(years),
        row("   Privatcars", "coach"),
        row("   Motorräder", "motorcycle", flag="8)"),
        row(moped_label, "moped", flag="8)"),
        ["5) Schnelle E-Bikes benötigen ein Motorfahrrad-Kontrollschild."] + empty,
    ]
    return pd.DataFrame(rows)


PASSENGER_VALUES = {
    "car": [100.0, 110.0],
    "coach": [1.0, 1.5],
    "motorcycle": [4.0, 5.0],
    "moped": [0.5, 0.25],
    "bus": [3.0, MISSING],
    "trolleybus": [0.25, MISSING],
}


@pytest.fixture
def freight_sheets():
    """Two freight sheets: recent with some flag columns, older with a lower header."""
    return {
        "2005-2006": freight_sheet(
            [2005.0, 2006.0, 2007],
            [20.0, 21.0, MISSING],
            [10.0, 11.0, MISSING],
            flag_years=[2005.0],
        ),
        "1989-2004": freight_sheet(
            [1989, 1990, 2004],
            [15.0, 16.0, 19.0],
            [7.0, 8.0, 9.0],
            header_row=3,
            label_suffix=" 3)",
        ),
        "1960-1980": freight_sheet([1960, 1980], [1.0, 2.0], [1.0, 2.0], header_row=3),
    }


@pytest.fixture
def passenger_sheets():
    """One passenger sheet for 2019-2020 with flag columns and missing bus values."""
    return {"2019-2020": passenger_sheet([2019, 2020], PASSENGER_VALUES)}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Leichte Fahrzeuge3", "Leichte Fahrzeuge"),
        ("Schwere Fahrzeuge 4)", "Schwere Fahrzeuge"),
        ("Autobusse 3)", "Autobusse"),
        ("   Personenwagen", "Personenwagen"),
        (
            "Motorfahrräder, inkl. schnelle E-Bikes 5)",
            "Motorfahrräder, inkl. schnelle E-Bikes",
        ),
        ("Strasse 2) 7), in Millionen", "Strasse 2) 7), in Millionen"),
        (
            "Total in- und ausländische Fahrzeuge 7)",
            "Total in- und ausländische Fahrzeuge",
        ),
        ("Personenwagen\nin Mio. Fahrzeug-km", "Personenwagen in Mio. Fahrzeug-km"),
        ("Total ", "Total"),
        (np.nan, ""),
        (None, ""),
    ],
)
def test_normalise_label(raw, expected):
    """Whitespace is collapsed and trailing footnote markers are dropped."""
    assert normalise_label(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (2005.0, 2005),
        (2025, 2025),
        ("1990", 1990),
        (np.nan, None),
        (None, None),
        ("T 11.5.1.1", None),
        (2005.5, None),
        (12.0, None),
        ("10)", None),
    ],
)
def test_parse_year(raw, expected):
    """Integer-valued numbers and four-digit strings are years; anything else is not."""
    assert parse_year(raw) == expected


def test_find_year_columns_skips_flag_columns(freight_sheets):
    """Flag columns between years are not mapped to years."""
    row, years = find_year_columns(freight_sheets["2005-2006"])
    assert row == 1
    assert years == {1: 2005, 3: 2006, 4: 2007}


def test_find_year_columns_finds_lower_header(freight_sheets):
    """Older sheets have the year header further down."""
    row, years = find_year_columns(freight_sheets["1989-2004"])
    assert row == 3
    assert sorted(years.values()) == [1989, 1990, 2004]


def test_find_year_columns_raises_without_years():
    """A sheet without a year header is rejected."""
    with pytest.raises(ValueError, match="No header row"):
        find_year_columns(pd.DataFrame([["a", "b"], ["c", 1.0]]))


def test_tidy_sheet_values_and_flags(freight_sheets):
    """Values become floats ('…' -> NaN) and flags are read from flag columns."""
    values, flags = tidy_sheet(freight_sheets["2005-2006"])
    assert list(values.columns) == [2005, 2006, 2007]
    assert values.loc["Leichte Fahrzeuge", 2006] == 21.0
    assert np.isnan(values.loc["Leichte Fahrzeuge", 2007])
    assert flags.loc["Schwere Fahrzeuge", 2005] == "r"
    assert flags.loc["Schwere Fahrzeuge", 2006] == ""


def test_select_rows_raises_on_missing_label(freight_sheets):
    """A required label that is missing is an error, not a silent gap."""
    values, _ = tidy_sheet(freight_sheets["2005-2006"])
    with pytest.raises(KeyError, match="Schwere Fahrzeuge"):
        select_rows(values.drop(index="Schwere Fahrzeuge"), FREIGHT_ROWS, "s")


def test_select_rows_raises_on_ambiguous_label(freight_sheets):
    """Two rows matching one component are rejected."""
    values, _ = tidy_sheet(freight_sheets["2005-2006"])
    doubled = pd.concat([values, values.loc[["Leichte Fahrzeuge"]]])
    with pytest.raises(ValueError, match="2 rows"):
        select_rows(doubled, FREIGHT_ROWS, "s")


def test_read_table_combines_sheets_and_drops_early_years(freight_sheets, caplog):
    """Sheets are stacked by year; years before 1990 and old sheets are dropped."""
    with caplog.at_level(logging.INFO):
        values, flags = read_table(freight_sheets, FREIGHT_ROWS, "freight")
    assert list(values.index) == [1990, 2004, 2005, 2006, 2007]
    assert values.loc[2004, "heavy"] == 9.0
    assert flags.loc[2005, "heavy"] == "r"
    assert "'1960-1980': skipped" in caplog.text
    assert "dropped years [1989]" in caplog.text


def test_read_table_rejects_overlapping_sheets(freight_sheets):
    """A year reported in two sheets is ambiguous."""
    sheets = {"a": freight_sheets["2005-2006"], "b": freight_sheets["2005-2006"]}
    with pytest.raises(ValueError, match="several sheets"):
        read_table(sheets, FREIGHT_ROWS, "freight")


def test_read_table_accepts_label_aliases():
    """Older passenger sheets label mopeds without the e-bike suffix."""
    values = {k: v[:1] for k, v in PASSENGER_VALUES.items()}
    sheets = {"x": passenger_sheet([2000], values, moped_label="   Motorfahrräder")}
    values, _ = read_table(sheets, PASSENGER_ROWS, "passenger")
    assert values.loc[2000, "moped"] == 0.5


def test_check_totals_passes_within_rounding(freight_sheets):
    """Small rounding differences pass; missing years are not checked."""
    values, _ = read_table(freight_sheets, FREIGHT_ROWS, "freight")
    values.loc[2005, "total"] *= 1 + 5e-4
    check_totals(values, "total", ["light", "heavy"], "freight")


def test_check_totals_raises_on_mismatch(passenger_sheets):
    """Components that do not add up to the total are an error."""
    values, _ = read_table(passenger_sheets, PASSENGER_ROWS, "passenger")
    values.loc[2020, "car"] += 1.0
    with pytest.raises(ValueError, match=r"does not sum to private_total.*\[2020\]"):
        check_totals(
            values, "private_total", ["car", "coach", "motorcycle", "moped"], "p"
        )


@pytest.fixture
def derived(freight_sheets, passenger_sheets):
    """Categories derived from the synthetic freight and passenger sheets."""
    freight, _ = read_table(freight_sheets, FREIGHT_ROWS, "freight")
    passenger, _ = read_table(passenger_sheets, PASSENGER_ROWS, "passenger")
    return derive_categories({"freight": freight, "passenger": passenger})


def _value(table, category, year, column="mvkm"):
    return table.query("vehicle_category == @category and year == @year")[column].item()


def test_derive_categories_sums_components(derived):
    """Motorcycles include mopeds; buses add trolleybuses and private coaches."""
    assert _value(derived, "passenger_car", 2019) == 100.0
    assert _value(derived, "motorcycle", 2019) == 4.5
    assert _value(derived, "bus", 2019) == 3.0 + 0.25 + 1.0
    assert _value(derived, "heavy_duty", 2004) == 9.0
    assert _value(derived, "light_duty", 2006) == 21.0


def test_derive_categories_leaves_incomplete_years_out(derived):
    """A category with a missing component is not derived for that year."""
    assert derived.query("vehicle_category == 'bus' and year == 2020").empty
    assert derived.query("vehicle_category == 'heavy_duty' and year == 2007").empty
    assert _value(derived, "passenger_car", 2020) == 110.0
    assert derived["mvkm"].notna().all()


def test_to_national_vkm_converts_and_labels(derived):
    """Values are converted from Mvkm to vkm and labelled."""
    national = to_national_vkm(derived, "CHE", "territorial", "test-source")
    assert _value(national, "heavy_duty", 2004, "vkm") == pytest.approx(9.0e6)
    assert set(national["country_id"]) == {"CHE"}
    assert set(national["unit"]) == {"vkm"}
    assert set(national["source"]) == {"test-source"}
    assert set(national["principle"]) == {"territorial"}
    assert national["year"].min() == 1990


def test_to_national_vkm_rejects_negative_values(derived):
    """Negative vehicle-km fail schema validation."""
    derived.loc[0, "mvkm"] = -1.0
    with pytest.raises(pandera.errors.SchemaError):
        to_national_vkm(derived, "CHE", "territorial", "s")


def _write_workbook(path, sheets):
    with pd.ExcelWriter(path) as writer:
        for name, sheet in sheets.items():
            sheet.to_excel(writer, sheet_name=name, header=False, index=False)
    return path


def test_main_reads_workbooks(tmp_path, freight_sheets, passenger_sheets):
    """End to end on synthetic Excel files: sheets, years and categories."""
    freight = _write_workbook(tmp_path / "freight.xlsx", freight_sheets)
    passenger = _write_workbook(tmp_path / "passenger.xlsx", passenger_sheets)
    out = tmp_path / "vkm.parquet"
    main(freight, passenger, {"CHE": "CH"}, {"CHE": "territorial"}, "s", out)
    national = pd.read_parquet(out)
    assert _value(national, "motorcycle", 2020, "vkm") == pytest.approx(5.25e6)
    assert len(national) == 4 + 4 + 2 + 2 + 1  # heavy, light, car, moto, bus


def test_main_raises_on_inconsistent_workbook(
    tmp_path, freight_sheets, passenger_sheets
):
    """An inconsistent total in the Excel file stops the import."""
    sheet = freight_sheets["2005-2006"]
    labels = sheet[0].map(normalise_label)
    sheet.loc[labels == "Total in- und ausländische Fahrzeuge", 3] = 100.0
    freight = _write_workbook(tmp_path / "freight.xlsx", freight_sheets)
    passenger = _write_workbook(tmp_path / "passenger.xlsx", passenger_sheets)
    with pytest.raises(ValueError, match=r"freight: .*\[2006\]"):
        main(
            freight,
            passenger,
            {"CHE": "CH"},
            {"CHE": "territorial"},
            "s",
            tmp_path / "vkm.parquet",
        )
