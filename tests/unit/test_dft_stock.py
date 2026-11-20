"""Unit tests for the DfT (Great Britain) road vehicle stock import."""

import logging

import numpy as np
import pandas as pd
import pandera.errors
import pytest
from prepare_dft_road_stock import (
    SHEET,
    check_totals,
    find_header_row,
    main,
    parse_quarter,
    read_licensed,
    select_year_end,
    to_national_stock,
)

COLUMNS = [
    "Geography",
    "Date",
    "Units",
    "Cars",
    "Motorcycles",
    "Light goods vehicles",
    "Heavy goods vehicles",
    "All goods vehicles",
    "Buses and coaches",
    "Other vehicles",
    "Total",
]
QUARTER_NAMES = {1: "March", 2: "June", 3: "September", 4: "December"}


def data_row(geography, year, quarter, cars, moto=1.0, lgv=2.0, hgv=0.5, bus=0.1):
    """One VEH0101a row in thousands; totals are consistent with the body types."""
    other = 0.25
    return [
        geography,
        f"{year} Q{quarter} (end {QUARTER_NAMES[quarter]})",
        "Thousands",
        cars,
        moto,
        lgv,
        hgv,
        lgv + hgv,
        bus,
        other,
        cars + moto + lgv + hgv + bus + other,
    ]


def licensed_sheet(rows):
    """Raw VEH0101a sheet as pandas reads it with `header=None`."""
    width = len(COLUMNS)
    title = [
        ["Licensed vehicles at the end of the quarter by body type"]
        + [None] * (width - 1),
        ["Table VEH0101a"] + [None] * (width - 1),
        ["Some shorthand is used in this table, [x] = not available."]
        + [None] * (width - 1),
    ]
    return pd.DataFrame(title + [COLUMNS] + rows)


@pytest.fixture
def sheet():
    """GB rows for 1989 Q4, 1994 Q3-Q4 (LGV not available in Q3), 2019 Q4, 2020 Q1.

    England rows have different values and must be ignored.
    """
    q3 = data_row("Great Britain", 1994, 3, 21.0)
    q3[5:8] = ["[x]", "[x]", 2.5]
    return licensed_sheet(
        [
            data_row("Great Britain", 1989, 4, 19.0),
            q3,
            data_row("Great Britain", 1994, 4, 21199.203, moto=720.922),
            data_row("England", 1994, 4, 18.0),
            data_row("Great Britain", 2019, 4, 31888.448, lgv=4123.356),
            data_row("Great Britain", 2020, 1, 32.0),
        ]
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2019 Q4 (end December)", (2019, 4)),
        ("1994 Q1 (end March)", (1994, 1)),
        (" 2026 Q2", (2026, 2)),
    ],
)
def test_parse_quarter(raw, expected):
    """Dates are read as year and quarter."""
    assert parse_quarter(raw) == expected


def test_parse_quarter_rejects_other_dates():
    """A date without a quarter fails loudly."""
    with pytest.raises(ValueError, match="cannot read date"):
        parse_quarter("December 2019")


def test_find_header_row(sheet):
    """The header row is found by its column titles."""
    assert find_header_row(sheet) == 3


def test_find_header_row_raises_without_header():
    """A sheet without the expected titles is rejected."""
    with pytest.raises(ValueError, match="No header row"):
        find_header_row(pd.DataFrame([["Geography", "Year", "Units"]]))


def test_read_licensed_converts_values(sheet):
    """Body type columns become floats; '[x]' becomes NaN."""
    table = read_licensed(sheet)
    assert len(table) == 6
    assert table.loc[1, "Cars"] == 21.0
    assert np.isnan(table.loc[1, "Light goods vehicles"])


def test_read_licensed_raises_on_missing_column(sheet):
    """A required body type column that is missing is an error."""
    renamed = sheet.replace("Buses and coaches", "Buses")
    with pytest.raises(KeyError, match="Buses and coaches"):
        read_licensed(renamed)


def test_select_year_end_keeps_q4_of_geography(sheet, caplog):
    """Only Q4 of the requested geography is kept, in vehicles."""
    with caplog.at_level(logging.WARNING):
        stock = select_year_end(read_licensed(sheet), "Great Britain")
    assert list(stock.index) == [1989, 1994, 2019]
    assert stock.loc[2019, "Cars"] == pytest.approx(31888448.0)
    assert stock.loc[1994, "Cars"] == pytest.approx(21199203.0)
    assert "no Q4 value for years [2020]" in caplog.text


def test_select_year_end_rejects_unknown_units(sheet):
    """Values in units other than thousands are not silently scaled."""
    table = read_licensed(sheet)
    table.loc[0, "Units"] = "Number"
    with pytest.raises(ValueError, match="unexpected units"):
        select_year_end(table, "Great Britain")


def test_select_year_end_rejects_duplicated_dates(sheet):
    """Two rows for the same geography and quarter are ambiguous."""
    table = read_licensed(sheet)
    table = pd.concat([table, table.iloc[[2]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated dates"):
        select_year_end(table, "Great Britain")


def test_select_year_end_raises_on_missing_geography(sheet):
    """A geography without rows fails loudly."""
    with pytest.raises(KeyError, match="Wales"):
        select_year_end(read_licensed(sheet), "Wales")


def test_check_totals_passes(sheet):
    """Consistent body types pass the total checks."""
    check_totals(select_year_end(read_licensed(sheet), "Great Britain"))


def test_check_totals_raises_on_mismatch(sheet):
    """Body types that do not add up to the total are an error."""
    stock = select_year_end(read_licensed(sheet), "Great Britain")
    stock.loc[2019, "All goods vehicles"] += 10.0
    with pytest.raises(ValueError, match=r"All goods vehicles in \[2019\]"):
        check_totals(stock)


def _value(table, category, year):
    return table.query("vehicle_category == @category and year == @year")[
        "stock"
    ].item()


def test_to_national_stock_maps_categories(sheet):
    """Body types map to categories; years before 1990 are dropped."""
    stock = select_year_end(read_licensed(sheet), "Great Britain")
    national = to_national_stock(stock, "GBR", "test-source")
    assert _value(national, "passenger_car", 2019) == 31888448.0
    assert _value(national, "light_duty", 2019) == 4123356.0
    assert _value(national, "motorcycle", 1994) == 720922.0
    assert _value(national, "heavy_duty", 1994) == 500.0
    assert _value(national, "bus", 2019) == 100.0
    assert sorted(national["year"].unique()) == [1994, 2019]
    assert set(national["source"]) == {"test-source"}
    assert set(national["unit"]) == {"vehicles"}


def test_to_national_stock_leaves_missing_values_out(sheet, caplog):
    """A missing body type value is left out and logged, not estimated."""
    stock = select_year_end(read_licensed(sheet), "Great Britain")
    stock.loc[2019, "Buses and coaches"] = np.nan
    with caplog.at_level(logging.WARNING):
        national = to_national_stock(stock, "GBR", "s")
    assert national.query("vehicle_category == 'bus' and year == 2019").empty
    assert len(national) == 2 * 5 - 1
    assert "GBR bus: no DfT value for years [2019]" in caplog.text


def test_to_national_stock_rejects_negative_values(sheet):
    """Negative stock fails schema validation."""
    stock = select_year_end(read_licensed(sheet), "Great Britain")
    stock.loc[2019, "Cars"] = -1.0
    with pytest.raises(pandera.errors.SchemaError):
        to_national_stock(stock, "GBR", "s")


def _write_ods(path, sheets):
    with pd.ExcelWriter(path, engine="odf") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, header=False, index=False)
    return path


def test_main_reads_ods(tmp_path, sheet):
    """End to end on a synthetic OpenDocument spreadsheet."""
    cover = pd.DataFrame([["Vehicles at the end of the quarter"]])
    ods = _write_ods(tmp_path / "veh0101.ods", {"Cover_sheet": cover, SHEET: sheet})
    out = tmp_path / "stock.parquet"
    main(ods, {"GBR": "GB"}, "s", out)
    national = pd.read_parquet(out)
    assert set(national["country_id"]) == {"GBR"}
    assert _value(national, "passenger_car", 2019) == 31888448.0


def test_main_raises_without_licensed_sheet(tmp_path, sheet):
    """A workbook without the licensed vehicles sheet is rejected."""
    ods = _write_ods(tmp_path / "veh0101.ods", {"Other": sheet})
    with pytest.raises(KeyError, match=SHEET):
        main(ods, {"GBR": "GB"}, "s", tmp_path / "stock.parquet")
