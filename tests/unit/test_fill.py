"""Unit tests for gap filling of the combined national tables (step 5)."""

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml
from _schemas import VEHICLE_CATEGORIES, FilledVkmSchema
from fill_national_tables import fill, interpolate_gaps, main, neighbour_ratios

MODULE_ROOT = Path(__file__).parents[2]


def vkm_table(years, values, source="s", principles=None):
    principles = principles or ["territorial"] * len(years)
    return pd.DataFrame(
        {
            "country_id": "AAA",
            "year": years,
            "vehicle_category": "bus",
            "vkm": [float(v) for v in values],
            "unit": "vkm",
            "source": source,
            "principle": principles,
        }
    )


def test_inner_gap_is_interpolated_linearly(caplog):
    with caplog.at_level(logging.INFO):
        result = fill(vkm_table([2000, 2004], [10, 50]), "vkm", max_years=3)
    assert result["year"].tolist() == [2000, 2001, 2002, 2003, 2004]
    assert result["vkm"].tolist() == [10, 20, 30, 40, 50]
    assert result["filled"].tolist() == ["observed"] + ["interpolated"] * 3 + ["observed"]
    assert set(result["source"]) == {"s"} and set(result["principle"]) == {"territorial"}
    assert "AAA bus 2002: vkm = 30 interpolated." in caplog.text


def test_long_gap_left_missing(caplog):
    with caplog.at_level(logging.WARNING):
        result = fill(vkm_table([2000, 2005], [10, 60]), "vkm", max_years=3)
    assert result["year"].tolist() == [2000, 2005]
    assert set(result["filled"]) == {"observed"}
    assert "years 2001-2004 not interpolated, gap of 4 years > 3" in caplog.text


def test_no_extrapolation_and_observed_unchanged():
    table = vkm_table([2001, 2002, 2004], [1, 2, 7])
    result = fill(table, "vkm", max_years=3)
    assert result["year"].tolist() == [2001, 2002, 2003, 2004]
    observed = result[result["filled"] == "observed"].drop(columns="filled")
    pd.testing.assert_frame_equal(observed.reset_index(drop=True), table)


def test_gap_between_different_principles_left_missing(caplog):
    table = vkm_table([2000, 2002], [1, 3], principles=["territorial", "nationality"])
    with caplog.at_level(logging.WARNING):
        result = interpolate_gaps(table, "vkm", max_years=3)
    assert len(result) == 2
    assert "differ" in caplog.text


def test_series_are_separate():
    """Gaps are filled per country and category, never across them."""
    other = vkm_table([2001], [100]).assign(vehicle_category="passenger_car")
    table = pd.concat([vkm_table([2000, 2002], [1, 3]), other], ignore_index=True)
    result = fill(table, "vkm", max_years=3)
    bus = result[result["vehicle_category"] == "bus"]
    assert bus["vkm"].tolist() == [1, 2, 3]
    assert (result["vehicle_category"] == "passenger_car").sum() == 1


def test_stock_and_main(tmp_path):
    stock = pd.DataFrame(
        {
            "country_id": "BBB",
            "year": [2010, 2012],
            "vehicle_category": "heavy_duty",
            "stock": [100.0, 120.0],
            "unit": "vehicles",
            "source": "s",
        }
    )
    stock.to_parquet(tmp_path / "in.parquet", index=False)
    main(tmp_path / "in.parquet", tmp_path / "out.parquet", "stock", 3, ["BBB"], {})
    out = pd.read_parquet(tmp_path / "out.parquet")
    assert out.set_index("year")["stock"].to_dict() == {2010: 100, 2011: 110, 2012: 120}
    assert out.columns[-1] == "filled"


def test_schema_rejects_unknown_fill_method():
    result = fill(vkm_table([2000], [1]), "vkm", max_years=3)
    with pytest.raises(Exception, match="filled"):
        FilledVkmSchema.validate(result.assign(filled="guessed"))


def stock_rows(country, category, years, values, source="s"):
    return pd.DataFrame(
        {
            "country_id": country,
            "year": years,
            "vehicle_category": category,
            "stock": [float(v) for v in values],
            "unit": "vehicles",
            "source": source,
        }
    )


def neighbour_table():
    """Target TTT has cars only; neighbours N1, N2 have cars and buses (2010-2011)."""
    return pd.concat(
        [
            stock_rows("TTT", "passenger_car", [2010, 2011, 2012], [1000, 1100, 1200]),
            stock_rows("N1", "passenger_car", [2010, 2011], [100, 100]),
            stock_rows("N1", "bus", [2010, 2011], [1, 2]),
            stock_rows("N2", "passenger_car", [2010, 2011], [200, 200]),
            stock_rows("N2", "bus", [2010, 2011], [6, 8]),
        ],
        ignore_index=True,
    )


def test_neighbour_ratios_mean_and_hold():
    ratios = neighbour_ratios(neighbour_table(), ["N1", "N2"], "bus", pd.Index([2010, 2011, 2012]))
    # 2010: (0.01 + 0.03) / 2; 2011: (0.02 + 0.04) / 2; 2012 holds 2011.
    assert ratios.round(6).tolist() == [0.02, 0.03, 0.03]


def test_missing_category_from_neighbour_ratio(caplog):
    with caplog.at_level(logging.INFO):
        result = fill(neighbour_table(), "stock", 3, ["TTT"], {"TTT": ["N1", "N2"]})
    assert set(result["country_id"]) == {"TTT"}  # neighbours are not written
    bus = result[result["vehicle_category"] == "bus"].set_index("year")
    assert bus["stock"].round(6).to_dict() == {2010: 20, 2011: 33, 2012: 36}
    assert set(bus["filled"]) == {"neighbour_ratio"}
    assert set(bus["source"]) == {"passenger_car stock x neighbour ratio (N1, N2)"}
    # Categories no neighbour has are left missing and logged.
    assert set(result["vehicle_category"]) == {"passenger_car", "bus"}
    assert "TTT heavy_duty: no neighbour ratio (left missing)." in caplog.text
    assert "TTT bus 2012: stock = 36" in caplog.text


def test_existing_categories_are_not_ratio_filled():
    """A category with some observed years is left to the other sub-steps."""
    table = pd.concat(
        [neighbour_table(), stock_rows("TTT", "bus", [2012], [50])], ignore_index=True
    )
    result = fill(table, "stock", 3, ["TTT"], {"TTT": ["N1", "N2"]})
    bus = result[result["vehicle_category"] == "bus"]
    assert bus["year"].tolist() == [2012] and bus["filled"].tolist() == ["observed"]


def test_no_car_stock_leaves_categories_missing(caplog):
    table = neighbour_table().query("country_id != 'TTT'")
    table = pd.concat([table, stock_rows("TTT", "bus", [2010], [5])], ignore_index=True)
    with caplog.at_level(logging.WARNING):
        result = fill(table, "stock", 3, ["TTT"], {"TTT": ["N1", "N2"]})
    assert len(result) == 1
    assert "TTT: no passenger_car stock" in caplog.text


def test_settings_neighbours_are_valid():
    """Neighbours are supported countries, not themselves filled from neighbours."""
    settings = yaml.safe_load(
        (MODULE_ROOT / "workflow/internal/settings.yaml").read_text(encoding="utf-8")
    )
    schema = yaml.safe_load((MODULE_ROOT / "workflow/internal/config.schema.yaml").read_text())
    supported = set(schema["properties"]["countries"]["items"]["enum"])
    neighbours = settings["gap_filling"]["neighbours"]
    assert set(neighbours) <= supported
    for country, names in neighbours.items():
        assert names and set(names) <= supported - set(neighbours), country
    assert {"ALB", "BIH", "MKD", "MNE", "SRB", "XKX", "ISL"} <= set(neighbours)
    assert set(neighbours["MKD"]) == {"BGR", "GRC", "HRV", "HUN", "ROU", "SVN"}
    assert len(VEHICLE_CATEGORIES) == 5
