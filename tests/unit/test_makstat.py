"""Unit tests for the MAKSTAT (North Macedonia) vehicle-km and vehicle stock import."""

import json
import logging

import pandas as pd
import pytest
from _makstat import log_updated, read_px_csv
from prepare_makstat_road_stock import (
    TYPES,
    check_total,
    national_by_year,
)
from prepare_makstat_road_stock import main as stock_main
from prepare_makstat_road_vkm import heavy_vehicle_km, to_national_vkm
from prepare_makstat_road_vkm import main as vkm_main

TOTAL_KM = "Kilometres travelled, in '000.000"
HIRE = "Of that, hire or reward, in '000.000 km"
OWN = "Of that, on own account, in '000.000 km"
LABEL = "REPUBLIC OF MACEDONIA"
FREIGHT_CSV = (
    '﻿"Passenger and freight transport","2003","2004","2019"\n'
    '"Kilometres travelled, in \'000",1,2,3\n'
    '"Freight transport 2)",..,..,..\n'
    f'"{TOTAL_KM}",287,496,835\n'
    f'"{HIRE}",0,277,254\n'
    f'"{OWN}",0,219,581\n'
)


def stock_csv(cars=(100, 120), total_offset=0):
    """Stock table with two years for the national row and one municipality."""
    header = ['"Municipalities"']
    national = [f'"{LABEL}"']
    for year, car in zip((2012, 2019), cars):
        values = {t: 1 for t in TYPES}
        values["passenger cars"] = car
        header += [f'"{year} Total"'] + [f'"{year} {t}"' for t in TYPES]
        national += [str(sum(values.values()) + total_offset)]
        national += [str(values[t]) for t in TYPES]
    other = ['"Skopje"'] + ["0"] * (len(header) - 1)
    return "﻿" + "\n".join(",".join(r) for r in (header, national, other)) + "\n"


@pytest.fixture
def freight_path(tmp_path):
    path = tmp_path / "freight.csv"
    path.write_text(FREIGHT_CSV, encoding="utf-8")
    return path


@pytest.fixture
def listing_path(tmp_path):
    path = tmp_path / "listing.json"
    entries = [
        {"id": "freight.px", "type": "t", "updated": "2026-05-27T12:00:00"},
        {"id": "stock.px", "type": "t", "updated": "2026-06-24T12:00:00"},
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_read_px_csv_strips_bom_and_missing(freight_path):
    """Row labels index the table; '..' becomes NaN."""
    table = read_px_csv(freight_path)
    assert list(table.columns) == ["2003", "2004", "2019"]
    assert table.loc["Freight transport 2)"].isna().all()
    assert table.loc[TOTAL_KM, "2019"] == 835


def test_log_updated(caplog, listing_path):
    """The table's update date from the folder listing is logged."""
    with caplog.at_level(logging.INFO):
        assert log_updated(listing_path, "stock.px") == "2026-06-24T12:00:00"
    assert "2026-06-24" in caplog.text
    with pytest.raises(KeyError):
        log_updated(listing_path, "other.px")


def test_heavy_km_drops_years_without_split(caplog, freight_path):
    """Years where total != hire + own account are dropped and logged."""
    with caplog.at_level(logging.WARNING):
        mvkm = heavy_vehicle_km(read_px_csv(freight_path))
    assert mvkm.to_dict() == {2004: 496, 2019: 835}
    assert "2003" in caplog.text


def test_heavy_km_requires_rows(freight_path):
    table = read_px_csv(freight_path).drop(index=OWN)
    with pytest.raises(KeyError, match="own account"):
        heavy_vehicle_km(table)


def test_vkm_to_national(freight_path):
    """Heavy-duty in vkm, with nationality principle and source."""
    national = to_national_vkm(
        heavy_vehicle_km(read_px_csv(freight_path)), "MKD", "nationality", "s"
    )
    assert set(national["vehicle_category"]) == {"heavy_duty"}
    assert national.set_index("year")["vkm"][2019] == pytest.approx(835e6)
    assert set(national["principle"]) == {"nationality"}


def test_vkm_main(tmp_path, freight_path, listing_path):
    out = tmp_path / "vkm.parquet"
    vkm_main(
        freight_path, listing_path, "freight.px", {"MKD": LABEL},
        {"MKD": "nationality"}, "s", out,
    )  # fmt: skip
    assert len(pd.read_parquet(out)) == 2


def test_national_by_year(tmp_path):
    """The national row is reshaped to year x vehicle type."""
    path = tmp_path / "stock.csv"
    path.write_text(stock_csv(), encoding="utf-8")
    wide = national_by_year(read_px_csv(path), LABEL)
    assert list(wide.index) == [2012, 2019]
    assert wide.loc[2019, "passenger cars"] == 120
    with pytest.raises(KeyError):
        national_by_year(read_px_csv(path), "Atlantis")


def test_check_total_raises(tmp_path):
    """A total that is not the sum of the vehicle types is an error."""
    path = tmp_path / "stock.csv"
    path.write_text(stock_csv(total_offset=5), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to the total"):
        check_total(national_by_year(read_px_csv(path), LABEL))


def test_stock_main_keeps_cars_and_buses(caplog, tmp_path, listing_path):
    """Only cars and buses are derived; the other categories are logged as missing."""
    path = tmp_path / "stock.csv"
    path.write_text(stock_csv(), encoding="utf-8")
    out = tmp_path / "stock.parquet"
    with caplog.at_level(logging.WARNING):
        stock_main(path, listing_path, "stock.px", {"MKD": LABEL}, "s", out)
    stock = pd.read_parquet(out)
    assert set(stock["vehicle_category"]) == {"passenger_car", "bus"}
    assert set(stock["country_id"]) == {"MKD"}
    cars = stock[stock["vehicle_category"] == "passenger_car"].set_index("year")
    assert cars["stock"].to_dict() == {2012: 100, 2019: 120}
    for category in ("heavy_duty", "light_duty", "motorcycle"):
        assert f"MKD {category}: not derived" in caplog.text
