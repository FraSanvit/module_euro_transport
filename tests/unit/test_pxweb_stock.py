"""Unit tests for unreliable-value exclusion and the Iceland and Kosovo stock imports."""

import json
import logging

import pandas as pd
import pytest
from _national import drop_excluded
from _pxweb import read_px_csv
from prepare_ask_road_stock import TYPES, check_total
from prepare_ask_road_stock import derive_categories as ask_derive
from prepare_ask_road_stock import main as ask_main
from prepare_hagstofa_road_stock import CATEGORY_COLUMNS
from prepare_hagstofa_road_stock import derive_categories as isl_derive


def stock_table():
    return pd.DataFrame(
        {
            "country_id": ["AAA"] * 4 + ["BBB"],
            "year": [2018, 2019, 2020, 2019, 2019],
            "vehicle_category": ["heavy_duty"] * 3 + ["bus", "heavy_duty"],
            "stock": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_drop_excluded_years_and_all(caplog):
    """Listed years or whole categories are dropped for that country only, and logged."""
    with caplog.at_level(logging.WARNING):
        kept = drop_excluded(
            stock_table(), {"AAA": {"heavy_duty": [2018, 2019], "bus": "all"}}
        )
    assert kept[["country_id", "year"]].values.tolist() == [["AAA", 2020], ["BBB", 2019]]
    assert "AAA heavy_duty: years [2018, 2019] excluded" in caplog.text


def test_drop_excluded_nothing():
    assert len(drop_excluded(stock_table(), {})) == 5
    assert len(drop_excluded(stock_table(), None)) == 5


def write(path, text):
    path.write_text("﻿" + text, encoding="utf-8")
    return path


def test_hagstofa_sums_classes(tmp_path):
    """Iceland categories sum EU classes; a missing part leaves the year out."""
    columns = [col for cols in CATEGORY_COLUMNS.values() for col in cols]
    header = '"Year",' + ",".join(f'"{c}"' for c in columns)
    row_2019 = '"2019",' + ",".join(str(i + 1) for i in range(len(columns)))
    row_2020 = '"2020",' + ",".join(".." if i == 2 else "1" for i in range(len(columns)))
    derived = isl_derive(read_px_csv(write(tmp_path / "s.csv", "\n".join([header, row_2019, row_2020]))))
    wide = derived.pivot(index="year", columns="vehicle_category", values="stock")
    # Columns: M1=1, M2=2, M3=3, N1=4, N2=5, N3=6, L*=7..11
    assert wide.loc[2019].to_dict() == {
        "bus": 5, "heavy_duty": 11, "light_duty": 4, "motorcycle": 45, "passenger_car": 1,
    }  # fmt: skip
    assert pd.isna(wide.loc[2020, "bus"])  # M3 missing in 2020
    assert wide.loc[2020, "passenger_car"] == 1


def test_hagstofa_requires_columns(tmp_path):
    path = write(tmp_path / "s.csv", '"Year","Vans (N1)"\n"2019",1\n')
    with pytest.raises(KeyError, match="M1"):
        isl_derive(read_px_csv(path))


def ask_csv(total_offset=0):
    """tr05-like table: types x years 2019, 2020."""
    values = {t: (i + 1) for i, t in enumerate(TYPES)}
    lines = ['"type of motor","2020","2019"']
    lines += [f'"{t}",{v},{v * 10}' for t, v in values.items()]
    total = sum(values.values())
    lines.append(f'"Total",{total + total_offset},{total * 10}')
    return "\n".join(lines) + "\n"


def test_ask_check_total(tmp_path):
    check_total(read_px_csv(write(tmp_path / "a.csv", ask_csv())))
    with pytest.raises(ValueError, match="2020"):
        check_total(read_px_csv(write(tmp_path / "b.csv", ask_csv(total_offset=3))))


def test_ask_categories(caplog, tmp_path):
    """Bus = buses + minibuses ("Van"); motorcycles are not derived."""
    with caplog.at_level(logging.WARNING):
        derived = ask_derive(read_px_csv(write(tmp_path / "a.csv", ask_csv())), "XKX")
    wide = derived.pivot(index="year", columns="vehicle_category", values="stock")
    assert wide.loc[2020].to_dict() == {
        "bus": 5 + 4, "heavy_duty": 2, "light_duty": 3, "passenger_car": 1,
    }  # fmt: skip
    assert "XKX motorcycle: not derived" in caplog.text


def test_ask_main_drops_excluded_years(tmp_path):
    listing = tmp_path / "listing.json"
    listing.write_text(json.dumps([{"id": "tr05.px", "updated": "2025-03-05"}]))
    out = tmp_path / "stock.parquet"
    ask_main(
        write(tmp_path / "a.csv", ask_csv()), listing, "tr05.px", {"XKX": "XK"}, "s",
        {"XKX": {"heavy_duty": [2019]}}, out,
    )  # fmt: skip
    stock = pd.read_parquet(out)
    heavy = stock[stock["vehicle_category"] == "heavy_duty"]
    assert heavy["year"].tolist() == [2020]
    assert len(stock) == 4 * 2 - 1
