"""Unit tests for gap filling of the combined national tables (step 5)."""

import logging

import pandas as pd
import pytest
from _schemas import FilledVkmSchema
from fill_national_tables import fill, interpolate_gaps, main


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
    main(tmp_path / "in.parquet", tmp_path / "out.parquet", "stock", 3)
    out = pd.read_parquet(tmp_path / "out.parquet")
    assert out.set_index("year")["stock"].to_dict() == {2010: 100, 2011: 110, 2012: 120}
    assert out.columns[-1] == "filled"


def test_schema_rejects_unknown_fill_method():
    result = fill(vkm_table([2000], [1]), "vkm", max_years=3)
    with pytest.raises(Exception, match="filled"):
        FilledVkmSchema.validate(result.assign(filled="guessed"))
