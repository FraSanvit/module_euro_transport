"""Unit tests: heavy_duty in the national vkm table comes from the heavy-duty level chain."""

import pandas as pd
import pytest
from combine_national_tables import combine, heavy_duty_rows, replace_heavy_duty


def source_table(country, category, years, vkm, source="JRC-IDEES-2021 v1", principle="unspecified"):
    return pd.DataFrame({"country_id": country, "year": years, "vehicle_category": category, "vkm": vkm,
                         "unit": "vkm", "source": source, "principle": principle})  # fmt: skip


def filled_levels():
    return pd.DataFrame(
        {
            "country_id": ["AAA", "AAA", "BBB"], "year": [2019, 2020, 2019], "total": [10.0, 11.0, 3.0],
            "filled_L_dom": ["observed", "observed", "neighbours_gdp_pps"],
            "filled_L_int": ["observed", "own_trend", "observed"],
            "filled_alpha_E": ["observed", "observed", "neighbour_mean"],
        }
    )  # fmt: skip


def test_heavy_duty_rows_are_territorial_totals_with_fill_methods():
    rows = heavy_duty_rows(filled_levels()).set_index(["country_id", "year"])
    assert rows.loc[("AAA", 2019), "source"] == "HDV level chain"
    assert rows.loc[("AAA", 2020), "source"] == "HDV level chain (filled: own_trend)"
    assert rows.loc[("BBB", 2019), "source"] == "HDV level chain (filled: neighbour_mean, neighbours_gdp_pps)"
    assert set(rows["principle"]) == {"territorial"} and rows.loc[("AAA", 2020), "vkm"] == 11


def test_replace_heavy_duty_drops_source_rows():
    frames = [
        pd.concat([source_table("AAA", "heavy_duty", [2000, 2019], [5.0, 6.0]),
                   source_table("AAA", "passenger_car", [2019], [100.0])])  # fmt: skip
    ]
    out = combine(replace_heavy_duty(frames, filled_levels(), ["AAA", "BBB"]))
    hdv = out[out["vehicle_category"] == "heavy_duty"].set_index(["country_id", "year"])["vkm"]
    assert hdv.to_dict() == pytest.approx({("AAA", 2019): 10.0, ("AAA", 2020): 11.0, ("BBB", 2019): 3.0})
    assert (out["vehicle_category"] == "passenger_car").sum() == 1
    only_aaa = combine(replace_heavy_duty(frames, filled_levels(), ["AAA"]))
    assert set(only_aaa["country_id"]) == {"AAA"}
