"""Unit tests for the BFS (Switzerland) road vehicle stock import."""

import logging

import numpy as np
import pandas as pd
import pandera.errors
import pytest
from prepare_bfs_road_stock import (
    check_group_totals,
    derive_categories,
    find_stock_header,
    main,
    parse_year_header,
    read_mopeds,
    read_stock,
    tidy_stock_sheet,
    to_national_stock,
)

GROUPS = {
    "Personenwagen": {"Personenwagen": 100.0, "Schwerer Personenwagen": 1.0},
    "Personentransportfahrzeuge 1": {
        "Leichter Motorwagen": 5.0,
        "Gesellschaftswagen": 2.0,
        "Kleinbus": 3.0,
        "Gelenkbus": 1.0,
    },
    "Sachentransportfahrzeuge": {
        "Lieferwagen": 20.0,
        "Lastwagen": 6.0,
        "Leichtes Sattelmotorfahrzeug": np.nan,
        "Schweres Sattelmotorfahrzeug": 0.5,
        "Sattelschlepper": 2.5,
    },
    "Landwirtschaftsfahrzeuge": {"Landwirtschaftlicher Traktor": 7.0},
    "Industriefahrzeuge": {"Traktor": 4.0},
    "Motorräder": {"Motorrad": 10.0, "Kleinmotorrad": 2.0, "Motorschlitten": 1.0},
    "Anhänger": {"Wohnanhänger": 9.0},
}
"""Group -> type -> stock in the first year; each later year adds 1 per type."""


def stock_sheet(years, total_label="Total Motorfahrzeuge (ohne Motorfahrräder)"):
    """Raw stock sheet as pandas reads it with `header=None`.

    Columns: group code, type code, group label, type label, then one per year.
    `years` are header cells (e.g. 2019.0 or '2026 p').
    """
    n = len(years)

    def values(base):
        return [base if np.isnan(base) else base + i for i in range(n)]

    def group_total(types):
        return [
            sum(0.0 if np.isnan(v) else v for v in vals)
            for vals in zip(*[values(b) for b in types.values()])
        ]

    motor = [g for g in GROUPS if g != "Anhänger"]
    totals = [group_total(GROUPS[g]) for g in motor]
    rows = [
        ["Strassenfahrzeugbestand nach Fahrzeuggruppe und Fahrzeugart", None, None]
        + [None] * (n + 1),
        ["FAZG", None, "Fahrzeuggruppe", None] + list(years),
        [None, "FAZ", None, "Fahrzeugart"] + [None] * n,
        ["1-6", None, total_label, None] + [sum(t) for t in zip(*totals)],
    ]
    for code, (group, types) in enumerate(GROUPS.items(), start=1):
        rows.append([code, None, group, None] + group_total(types))
        for kind, base in types.items():
            rows.append([None, 10 * code, None, kind] + values(base))
    rows += [
        ["1 ohne Trolleybusse und Gelenktrolleybusse"] + [None] * (n + 3),
        ["Bestand am 30. September"] + [None] * (n + 3),
    ]
    return pd.DataFrame(rows)


def moped_sheet(years, totals, split_from=None):
    """Raw moped sheet: years in one header row, a sub-header row, regions below.

    From `split_from` on, each year has a second 'davon schnelle E-Bikes' column.
    """
    header, sub, total_row, region_row = [None], [None], ["Total 1"], ["Bern"]
    for year, total in zip(years, totals):
        split = split_from is not None and year >= split_from
        header.append(year)
        sub.append("Total" if split else None)
        total_row.append(total)
        region_row.append(total / 2)
        if split:
            header.append(None)
            sub.append("davon schnelle \nE-Bikes")
            total_row.append("…")
            region_row.append(1.0)
    width = len(header)
    rows = [
        ["Bestand der Motorfahrräder nach Kanton"] + [None] * (width - 1),
        header,
        sub,
        total_row,
        region_row,
        ["Hinweis: Stichtag ist in der Regel 31. Dezember"] + [None] * (width - 1),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def stock_sheets():
    """A recent sheet with footnote and provisional years and an older sheet."""
    return {
        "2018-2020": stock_sheet(["2018 2", 2019.0, "2020 p"]),
        "1989-2017": stock_sheet([1989, 2017.0], total_label="Total Motorfahrzeuge"),
    }


@pytest.fixture
def moped_sheets():
    """Moped totals 1985-2019, with e-bike columns from 2018."""
    sheet = moped_sheet([1985, 1990, 2017, 2018, 2019], [9, 8, 7, 6, 5], 2018)
    return {"1985-2019": sheet}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (2019.0, (2019, "")),
        ("2018 2", (2018, "2")),
        ("2026 p", (2026, "p")),
        ("1990", (1990, "")),
        ("Fahrzeuggruppe", None),
        ("T11.3.2.1.3", None),
        (np.nan, None),
    ],
)
def test_parse_year_header(raw, expected):
    """Year cells may carry a footnote marker after a space."""
    assert parse_year_header(raw) == expected


def test_find_stock_header_by_column_titles(stock_sheets):
    """The header row and label columns are found by their titles."""
    row, group_col, type_col, years = find_stock_header(stock_sheets["2018-2020"])
    assert (row, group_col, type_col) == (1, 2, 3)
    assert years == {4: (2018, "2"), 5: (2019, ""), 6: (2020, "p")}


def test_find_stock_header_raises_without_titles():
    """A sheet without the group/type titles is rejected."""
    with pytest.raises(ValueError, match="Fahrzeugart"):
        find_stock_header(pd.DataFrame([["x", "Fahrzeuggruppe", 2019.0]]))


def test_tidy_stock_sheet_assigns_types_to_groups(stock_sheets):
    """Type rows are keyed by their group; footnote rows are skipped."""
    values, markers = tidy_stock_sheet(stock_sheets["2018-2020"])
    assert values.loc[("Personenwagen", ""), 2019] == 103.0
    assert values.loc[("Personentransportfahrzeuge", "Kleinbus"), 2018] == 3.0
    assert np.isnan(
        values.loc[("Sachentransportfahrzeuge", "Leichtes Sattelmotorfahrzeug"), 2019]
    )
    assert markers == {2018: "2", 2019: "", 2020: "p"}
    assert "Bestand am 30. September" not in values.index.get_level_values("group")


def test_read_stock_drops_provisional_and_early_years(stock_sheets, caplog):
    """Provisional years and years before 1990 are dropped and logged."""
    with caplog.at_level(logging.INFO):
        stock = read_stock(stock_sheets)
    assert list(stock.columns) == [2017, 2018, 2019]
    assert "2020 is provisional, dropped" in caplog.text
    assert "dropped years [1989]" in caplog.text
    assert "2018 has footnote 2" in caplog.text


def test_read_stock_rejects_overlapping_sheets(stock_sheets):
    """A year reported in two sheets is ambiguous."""
    sheets = {"a": stock_sheets["2018-2020"], "b": stock_sheets["2018-2020"]}
    with pytest.raises(ValueError, match="several sheets"):
        read_stock(sheets)


def test_check_group_totals_passes(stock_sheets):
    """Consistent groups pass; missing type values count as zero."""
    values, _ = tidy_stock_sheet(stock_sheets["2018-2020"])
    check_group_totals(values, "s")


def test_check_group_totals_raises_on_group_mismatch(stock_sheets):
    """A type that does not add up to its group total is an error."""
    values, _ = tidy_stock_sheet(stock_sheets["2018-2020"])
    values.loc[("Sachentransportfahrzeuge", "Lastwagen"), 2019] += 1
    with pytest.raises(ValueError, match=r"'Sachentransportfahrzeuge'.*\[2019\]"):
        check_group_totals(values, "s")


def test_check_group_totals_raises_on_motor_vehicle_total(stock_sheets):
    """The motor vehicle total must equal the sum of its groups."""
    values, _ = tidy_stock_sheet(stock_sheets["1989-2017"])
    values.loc[("Total Motorfahrzeuge", ""), 2017] += 1
    with pytest.raises(ValueError, match=r"'Total Motorfahrzeuge'.*\[2017\]"):
        check_group_totals(values, "s")


def test_check_group_totals_raises_without_total(stock_sheets):
    """A missing motor vehicle total row fails loudly."""
    values, _ = tidy_stock_sheet(stock_sheets["2018-2020"])
    values = values.drop(index=("Total Motorfahrzeuge (ohne Motorfahrräder)", ""))
    with pytest.raises(KeyError, match="expected one row"):
        check_group_totals(values, "s")


def test_read_mopeds_reads_total_row(moped_sheets):
    """The national total is read per year from 1990; e-bike columns are ignored."""
    mopeds = read_mopeds(moped_sheets)
    assert mopeds.to_dict() == {1990: 8.0, 2017: 7.0, 2018: 6.0, 2019: 5.0}


def test_read_mopeds_rejects_other_measures(moped_sheets):
    """A year column whose sub-header is not the total is rejected."""
    sheet = moped_sheets["1985-2019"]
    sheet.iat[2, 4] = "davon schnelle E-Bikes"  # below 2018
    with pytest.raises(ValueError, match="unexpected sub-header"):
        read_mopeds(moped_sheets)


def test_read_mopeds_raises_without_total(moped_sheets):
    """A missing national total row fails loudly."""
    moped_sheets["1985-2019"].iat[3, 0] = "Schweiz"
    with pytest.raises(KeyError, match="Total"):
        read_mopeds(moped_sheets)


@pytest.fixture
def derived(stock_sheets, moped_sheets):
    """Categories derived from the synthetic stock and moped sheets."""
    return derive_categories(read_stock(stock_sheets), read_mopeds(moped_sheets))


def _value(table, category, year):
    return table.query("vehicle_category == @category and year == @year")[
        "stock"
    ].item()


def test_derive_categories_maps_rows(derived):
    """Cars use the group total; buses, heavy duty and motorcycles sum rows."""
    # 2019 is the second year of its sheet: every type is base + 1.
    assert _value(derived, "passenger_car", 2019) == 101.0 + 2.0
    assert _value(derived, "bus", 2019) == 3.0 + 4.0 + 2.0
    assert _value(derived, "light_duty", 2019) == 21.0
    assert _value(derived, "heavy_duty", 2019) == 7.0 + 1.5 + 3.5
    assert _value(derived, "motorcycle", 2019) == 11.0 + 3.0 + 2.0 + 5.0


def test_derive_categories_requires_mopeds_for_motorcycles(stock_sheets, caplog):
    """Motorcycles are left out for years without a moped total."""
    mopeds = pd.Series({2019: 5.0}).rename_axis("year")
    with caplog.at_level(logging.WARNING):
        derived = derive_categories(read_stock(stock_sheets), mopeds)
    assert sorted(derived.query("vehicle_category == 'motorcycle'")["year"]) == [2019]
    assert len(derived.query("vehicle_category == 'bus'")) == 3
    assert "motorcycle: incomplete BFS values for years [2017, 2018]" in caplog.text


def test_derive_categories_raises_on_missing_row(stock_sheets, moped_sheets):
    """A required row that is missing is an error, not a silent gap."""
    stock = read_stock(stock_sheets)
    stock = stock.drop(index=("Personentransportfahrzeuge", "Gelenkbus"))
    with pytest.raises(KeyError, match="Gelenkbus"):
        derive_categories(stock, read_mopeds(moped_sheets))


def test_to_national_stock_labels(derived):
    """The output follows the national stock schema."""
    national = to_national_stock(derived, "CHE", "test-source")
    assert set(national["country_id"]) == {"CHE"}
    assert set(national["unit"]) == {"vehicles"}
    assert set(national["source"]) == {"test-source"}
    assert len(national) == 5 * 3


def test_to_national_stock_rejects_negative_values(derived):
    """Negative stock fails schema validation."""
    derived.loc[0, "stock"] = -1.0
    with pytest.raises(pandera.errors.SchemaError):
        to_national_stock(derived, "CHE", "s")


def _write_workbook(path, sheets):
    with pd.ExcelWriter(path) as writer:
        for name, sheet in sheets.items():
            sheet.to_excel(writer, sheet_name=name, header=False, index=False)
    return path


def test_main_reads_workbooks(tmp_path, stock_sheets, moped_sheets):
    """End to end on synthetic Excel files."""
    stock = _write_workbook(tmp_path / "stock.xlsx", stock_sheets)
    mopeds = _write_workbook(tmp_path / "mopeds.xlsx", moped_sheets)
    out = tmp_path / "stock.parquet"
    main(stock, mopeds, {"CHE": "CH"}, "s", out)
    national = pd.read_parquet(out)
    assert sorted(national["year"].unique()) == [2017, 2018, 2019]
    assert _value(national, "motorcycle", 2017) == 11.0 + 3.0 + 2.0 + 7.0


def test_main_raises_on_inconsistent_workbook(tmp_path, stock_sheets, moped_sheets):
    """An inconsistent group total in the Excel file stops the import."""
    sheet = stock_sheets["2018-2020"]
    sheet.loc[sheet[2] == "Motorräder", 5] = 1.0
    stock = _write_workbook(tmp_path / "stock.xlsx", stock_sheets)
    mopeds = _write_workbook(tmp_path / "mopeds.xlsx", moped_sheets)
    with pytest.raises(ValueError, match=r"'Motorräder'.*\[2019\]"):
        main(stock, mopeds, {"CHE": "CH"}, "s", tmp_path / "stock.parquet")
