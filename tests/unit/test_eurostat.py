"""Unit tests for the Eurostat road vehicle-km import."""

import logging

import pandas as pd
import pytest
from prepare_eurostat_road_vkm import (
    derive_categories,
    log_flags,
    read_sdmx_csv,
    to_national_vkm,
)

HEADER = "DATAFLOW,LAST UPDATE,freq,{dim},unit,vehicle,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG,CONF_STATUS"


def write_sdmx(path, dim, rows):
    """Write a minimal SDMX-CSV file. Rows: (dim value, unit, vehicle, geo, year, value, flag)."""
    lines = [HEADER.format(dim=dim)]
    for dim_value, unit, vehicle, geo, year, value, flag in rows:
        lines.append(
            f"ESTAT:X(1.0),01/01/26 23:00:00,A,{dim_value},{unit},{vehicle},{geo},{year},{value},{flag},"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def vehmov_path(tmp_path):
    """Synthetic road_tf_vehmov: AA complete, NO without motorcycles."""
    rows = [
        ("TERNAT_REG", "MIO_VKM", "CAR", "AA", 2019, 100, ""),
        ("TERNAT_REG", "MIO_VKM", "BUS_MCO_TRO", "AA", 2019, 5, ""),
        ("TERNAT_REG", "MIO_VKM", "MOTO_MOP", "AA", 2019, 3, "e"),
        ("TERNAT_REG", "MIO_VKM", "LOR_RTRN", "AA", 2019, 42, ""),
        ("TERNAT_REGNAT", "MIO_VKM", "CAR", "AA", 2019, 90, ""),  # other principle: ignored
        ("TERNAT_REG", "MIO_VKM", "CAR", "NO", 2019, 50, ""),
        ("TERNAT_REG", "MIO_VKM", "BUS_MCO_TRO", "NO", 2019, 1, ""),
        ("TERNAT_REG", "MIO_VKM", "CAR", "NO", 2020, "", ":"),  # missing value
    ]
    return write_sdmx(tmp_path / "vehmov.csv", "regisveh", rows)


@pytest.fixture
def road_path(tmp_path):
    """Synthetic road_tf_road: AA with full lorry split, NO without road tractors."""
    rows = [
        ("TOTAL", "MIO_VKM", "LOR", "AA", 2019, 30, ""),
        ("TOTAL", "MIO_VKM", "LOR_LE3P5", "AA", 2019, 22, ""),
        ("TOTAL", "MIO_VKM", "TRC", "AA", 2019, 10, ""),
        ("MWAY", "MIO_VKM", "LOR", "AA", 2019, 12, ""),  # motorway only: ignored
        ("TOTAL", "MIO_VKM", "LOR", "NO", 2019, 9, ""),
        ("TOTAL", "MIO_VKM", "LOR_LE3P5", "NO", 2019, 7, ""),
    ]
    return write_sdmx(tmp_path / "road.csv", "tra_infr", rows)


@pytest.fixture
def derived(vehmov_path, road_path):
    """Derived categories from both synthetic files."""
    vehmov = read_sdmx_csv(vehmov_path, "regisveh", "TERNAT_REG")
    road = read_sdmx_csv(road_path, "tra_infr", "TOTAL")
    return derive_categories(vehmov, road)


def value(table, geo, category, year=2019, column="mvkm"):
    """Pick one value from a long table."""
    key = "geo" if "geo" in table else "country_id"
    row = table[
        (table[key] == geo)
        & (table["vehicle_category"] == category)
        & (table["year"] == year)
    ]
    return row[column].item()


def test_read_filters_dimension_and_drops_missing(vehmov_path):
    """Only the requested dimension value is kept, and empty values are dropped."""
    vehmov = read_sdmx_csv(vehmov_path, "regisveh", "TERNAT_REG")
    assert set(vehmov["year"]) == {2019}
    assert vehmov.query("geo == 'AA' and vehicle == 'CAR'")["mvkm"].item() == 100


def test_read_keeps_geo_no_as_string(vehmov_path):
    """Norway's geo code "NO" must not be parsed as missing or boolean."""
    vehmov = read_sdmx_csv(vehmov_path, "regisveh", "TERNAT_REG")
    assert "NO" in set(vehmov["geo"])


def test_read_rejects_wrong_layout(tmp_path):
    """A file without the expected dimension column is rejected."""
    path = write_sdmx(tmp_path / "x.csv", "other", [])
    with pytest.raises(ValueError, match="missing columns"):
        read_sdmx_csv(path, "regisveh", "TERNAT_REG")


def test_derive_heavy_and_light_split(derived):
    """heavy = LOR - LOR_LE3P5 + TRC; light = LOR_LE3P5."""
    assert value(derived, "AA", "heavy_duty") == pytest.approx(30 - 22 + 10)
    assert value(derived, "AA", "light_duty") == pytest.approx(22)


def test_derive_passenger_categories(derived):
    """Cars, buses and motorcycles come from road_tf_vehmov."""
    assert value(derived, "AA", "passenger_car") == 100
    assert value(derived, "AA", "bus") == 5
    assert value(derived, "AA", "motorcycle") == 3


def test_derive_no_heavy_without_road_tractors(derived):
    """Without road tractors, heavy-duty is not derived (no estimation)."""
    no = derived[derived["geo"] == "NO"]
    assert "heavy_duty" not in set(no["vehicle_category"])
    assert "light_duty" in set(no["vehicle_category"])
    assert "motorcycle" not in set(no["vehicle_category"])


def test_derive_rejects_negative_heavy():
    """Lorries <= 3.5 t larger than all lorries plus tractors is a data error."""
    road = pd.DataFrame(
        {
            "geo": ["AA"] * 3,
            "vehicle": ["LOR", "LOR_LE3P5", "TRC"],
            "year": [2019] * 3,
            "mvkm": [10.0, 20.0, 1.0],
            "flag": [""] * 3,
        }
    )
    vehmov = road.iloc[:0]
    with pytest.raises(ValueError, match="Negative"):
        derive_categories(vehmov, road)


def test_lorry_consistency_warning(caplog, vehmov_path, road_path):
    """LOR + TRC (40) vs LOR_RTRN (42) differ by 5%: a warning is logged."""
    vehmov = read_sdmx_csv(vehmov_path, "regisveh", "TERNAT_REG")
    road = read_sdmx_csv(road_path, "tra_infr", "TOTAL")
    with caplog.at_level(logging.WARNING):
        derive_categories(vehmov, road)
    assert "LOR_RTRN" in caplog.text


def test_to_national_selects_converts_and_labels(derived):
    """Only requested countries are kept, in vkm, with source and principle."""
    national = to_national_vkm(
        derived, {"NOR": "NO"}, {"NOR": "territorial"}, "Eurostat test"
    )
    assert set(national["country_id"]) == {"NOR"}
    assert value(national, "NOR", "passenger_car", column="vkm") == pytest.approx(50e6)
    assert set(national["principle"]) == {"territorial"}
    assert set(national["source"]) == {"Eurostat test"}


def test_to_national_logs_missing_categories(caplog, derived):
    """Missing categories are logged, not filled."""
    with caplog.at_level(logging.WARNING):
        to_national_vkm(derived, {"NOR": "NO"}, {"NOR": "territorial"}, "s")
    assert "heavy_duty" in caplog.text and "motorcycle" in caplog.text


def test_to_national_raises_for_country_without_data(derived):
    """A requested country with no data at all is an error."""
    with pytest.raises(ValueError, match="ZZZ"):
        to_national_vkm(derived, {"ZZZ": "ZZ"}, {"ZZZ": "territorial"}, "s")


def test_to_national_requires_principle(derived):
    """Every country needs an explicit accounting principle."""
    with pytest.raises(ValueError, match="principle"):
        to_national_vkm(derived, {"NOR": "NO"}, {}, "s")


def test_log_flags(caplog, vehmov_path):
    """Flagged observations used for selected countries are logged."""
    vehmov = read_sdmx_csv(vehmov_path, "regisveh", "TERNAT_REG")
    with caplog.at_level(logging.INFO):
        log_flags(vehmov, ["AA"], "road_tf_vehmov")
    assert "flagged 'e'" in caplog.text
