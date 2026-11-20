"""Unit tests for the Eurostat vehicle stock import."""

import logging

import pytest
from prepare_eurostat_road_stock import (
    derive_categories,
    read_stock_tables,
    to_national_stock,
)

HEADERS = {
    "road_eqs_lorroa": ["freq", "unit", "vehicle", "age"],
    "road_eqs_carmot": ["freq", "unit", "mot_nrg", "engine"],
    "road_eqs_busveh": ["freq", "vehicle", "unit"],
    "road_eqs_mopeds": ["freq", "mot_nrg", "vehicle", "unit"],
}


def write_sdmx(path, dims, rows):
    """Write a minimal SDMX-CSV. Rows: (dict of dimension values, geo, year, value)."""
    header = ["DATAFLOW", "LAST UPDATE", *dims, "geo", "TIME_PERIOD", "OBS_VALUE"]
    header += ["OBS_FLAG", "CONF_STATUS"]
    lines = [",".join(header)]
    for values, geo, year, value in rows:
        cells = [values.get(d, "A" if d == "freq" else "") for d in dims]
        lines.append(",".join(["ESTAT:X(1.0)", "01/01/26", *cells, geo, str(year), str(value), "", ""]))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def paths(tmp_path):
    """Synthetic stock datasets for AA (complete), NO (no tractors) and MK."""
    lor = lambda v, a="TOTAL": {"unit": "NR", "vehicle": v, "age": a}  # noqa: E731
    lorroa = [
        (lor("LOR_GT3P5"), "AA", 2019, 100),
        (lor("TRC"), "AA", 2019, 20),
        (lor("VG_LE3P5"), "AA", 2019, 500),
        (lor("LOR_GT3P5", "Y_LT2"), "AA", 2019, 9),  # age class: ignored
        (lor("LOR_GT3P5"), "NO", 2019, 50),
        (lor("VG_LE3P5"), "NO", 2019, 300),
        (lor("LOR_GT3P5"), "MK", 2019, 38),
        (lor("TRC"), "MK", 2019, 5),
        (lor("LOR_GT3P5"), "AA", 1985, 90),  # before 1990: dropped
        (lor("TRC"), "AA", 1985, 10),
    ]
    car = {"unit": "NR", "mot_nrg": "TOTAL", "engine": "TOTAL"}
    carmot = [
        (car, "AA", 2019, 4000),
        (car, "NO", 2019, 2800),
        ({**car, "mot_nrg": "DIE"}, "AA", 2019, 1),  # not total: ignored
    ]
    busveh = [({"unit": "NR", "vehicle": "TOTAL"}, "AA", 2019, 15)]
    mop = lambda v: {"unit": "NR", "mot_nrg": "TOTAL", "vehicle": v}  # noqa: E731
    mopeds = [
        (mop("MOTO"), "AA", 2019, 70),
        (mop("MOP"), "AA", 2019, 30),
        (mop("MOTO"), "NO", 2019, 200),  # no mopeds for NO: no motorcycle value
    ]
    data = {
        "road_eqs_lorroa": lorroa,
        "road_eqs_carmot": carmot,
        "road_eqs_busveh": busveh,
        "road_eqs_mopeds": mopeds,
    }
    return {
        name: write_sdmx(tmp_path / f"{name}.csv", HEADERS[name], rows)
        for name, rows in data.items()
    }


@pytest.fixture
def derived(paths):
    """Derived categories from the synthetic datasets."""
    return derive_categories(read_stock_tables(paths))


def value(table, geo, category, year=2019):
    """Pick one stock value."""
    key = "geo" if "geo" in table else "country_id"
    row = table[
        (table[key] == geo)
        & (table["vehicle_category"] == category)
        & (table["year"] == year)
    ]
    return row["stock"].item() if len(row) else None


def test_heavy_is_lorries_plus_tractors(derived):
    """heavy_duty = lorries > 3.5 t + road tractors (age total only)."""
    assert value(derived, "AA", "heavy_duty") == 120


def test_heavy_needs_tractors(derived):
    """Without road tractors, heavy-duty is not derived."""
    assert value(derived, "NO", "heavy_duty") is None
    assert value(derived, "NO", "light_duty") == 300


def test_motorcycle_needs_mopeds(derived):
    """motorcycle = motorcycles + mopeds; missing mopeds leaves it missing."""
    assert value(derived, "AA", "motorcycle") == 100
    assert value(derived, "NO", "motorcycle") is None


def test_car_and_bus_totals(derived):
    """Cars use the all-fuel, all-engine total; buses the vehicle total."""
    assert value(derived, "AA", "passenger_car") == 4000
    assert value(derived, "AA", "bus") == 15


def test_to_national_maps_codes_and_drops_old_years(derived):
    """ISO codes are mapped (incl. "NO"), pre-1990 years are dropped."""
    stock = to_national_stock(derived, {"AAA": "AA", "NOR": "NO"}, "s")
    assert set(stock["country_id"]) == {"AAA", "NOR"}
    assert stock["year"].min() == 2019
    assert set(stock["unit"]) == {"vehicles"}


def test_to_national_logs_missing_categories(caplog, derived):
    """Categories without data are logged as missing."""
    with caplog.at_level(logging.WARNING):
        to_national_stock(derived, {"NOR": "NO"}, "s")
    assert "NOR heavy_duty: no Eurostat stock" in caplog.text
    assert "NOR bus: no Eurostat stock" in caplog.text
