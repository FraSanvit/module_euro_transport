"""Unit tests for the heavy-duty gap filling (MULTIYEAR_PLAN.md §6.9)."""

import logging

import numpy as np
import pandas as pd
import pytest
from _heavy_duty_fill import drop_outliers, fill_levels, fill_series, theil_sen

YEARS = list(range(2010, 2016))


def test_theil_sen_ignores_a_single_break():
    slope, intercept = theil_sen([1, 2, 3, 4, 5], [1.0, 2, 3, 4, 50])
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(0.0)
    assert theil_sen([3], [7.0]) == (0.0, 7.0)  # one point: held


def test_fill_series_inner_edges_and_clipping(caplog):
    s = pd.Series({2011: 1.0, 2012: 2.0, 2015: 5.0})
    with caplog.at_level(logging.WARNING):
        values, labels = fill_series(s, YEARS, max_inner=1, window=5, name="AA x")
    assert values[2013] == pytest.approx(3) and values[2014] == pytest.approx(4)
    assert labels[2013] == "interpolated"
    assert "gap 2013-2014 of 2 years > 1 interpolated" in caplog.text
    # Start: trend gives 0 in 2010, clipped to the minimum of the observed window (1).
    assert values[2010] == pytest.approx(1.0) and labels[2010] == "own_trend"
    rising = pd.Series({2010: 1.0, 2011: 2.0, 2012: 3.0})
    values, labels = fill_series(rising, YEARS, window=3)
    assert values[2015] == pytest.approx(3.0)  # trend 6 clipped to the window's maximum
    assert (labels[[2013, 2014, 2015]] == "own_trend").all()
    empty, labels = fill_series(pd.Series(dtype=float), YEARS)
    assert empty.isna().all() and (labels == "").all()


def test_drop_outliers():
    s = pd.Series({2010: 0.02, 2011: 0.24, 2012: 0.80, 2013: 0.81, 2014: 0.79, 2015: 0.80})
    clean, bad = drop_outliers(s, n_mad=4, min_abs=0.2)
    assert bad[[2010, 2011]].all() and not bad[2012:].any()
    modest = pd.Series({2010: 0.61, 2011: 0.60, 2012: 0.73, 2013: 0.73, 2014: 0.72, 2015: 0.74})
    assert not drop_outliers(modest, n_mad=4, min_abs=0.2)[1].any()  # a step of 0.12 is kept


def level_rows(country, l_dom, l_int, alpha, t=None, foreign=None, own=None):
    n = len(YEARS)
    as_list = lambda v: list(v) if isinstance(v, (list, tuple)) else [v] * n  # noqa: E731
    return pd.DataFrame(
        {
            "country_id": country, "year": YEARS, "L_dom": as_list(l_dom), "L_int": as_list(l_int),
            "alpha_E": as_list(alpha), "T": as_list(t if t is not None else np.nan),
            "L_int_foreign": as_list(foreign if foreign is not None else np.nan),
            "L_int_own": as_list(own if own is not None else np.nan), "CAB": 0.0,
        }
    )  # fmt: skip


def test_fill_levels_all_cases():
    nan = np.nan
    levels = pd.concat(
        [
            level_rows("NA1", 100.0, 50.0, 0.8),  # neighbours, complete
            level_rows("NA2", 300.0, 30.0, 0.6),
            level_rows("TTT", nan, 10.0, nan),  # no domestic series: GDP PPS x neighbours
            level_rows("KKK", nan, nan, nan),  # no PPS, no international: GDP EUR, ratios
            level_rows("GBR", [80, 80, nan, nan, nan, nan], [20, 20, nan, nan, nan, nan], 0.9,
                       t=[100, 100, 90, 90, 90, 90]),  # territorial total known, split missing
            level_rows("IRL", 100.0, [40, 40, 40, nan, nan, nan], 0.8,
                       foreign=[30, 30, 30, nan, nan, nan], own=10.0),  # foreign part missing
        ]
    )  # fmt: skip
    gdp = pd.DataFrame(
        [{"country_id": c, "year": y, "gdp_pps": pps, "gdp_eur": eur}
         for c, pps, eur in [("NA1", 10.0, 20.0), ("NA2", 10.0, 20.0), ("TTT", 2.0, 2.0), ("KKK", nan, 4.0),
                             ("GBR", 50.0, 50.0), ("IRL", 10.0, 10.0)]
         for y in YEARS]  # fmt: skip
    )
    neighbours = {"TTT": ["NA1", "NA2"], "KKK": ["NA1", "NA2"]}
    out = fill_levels(levels, gdp, YEARS, neighbours).set_index(["country_id", "year"])
    ttt = out.loc[("TTT", 2012)]
    assert ttt["L_dom"] == pytest.approx(2 * (10 + 30) / 2)  # GDP PPS 2 x mean D/GDP (10, 30)
    assert ttt["filled_L_dom"] == "neighbours_gdp_pps"
    assert ttt["alpha_E"] == pytest.approx(0.7) and ttt["filled_alpha_E"] == "neighbour_mean"
    assert ttt["L_int"] == 10 and ttt["filled_L_int"] == "observed"
    kkk = out.loc[("KKK", 2012)]
    assert kkk["L_dom"] == pytest.approx(4 * (5 + 15) / 2) and kkk["filled_L_dom"] == "neighbours_gdp_eur"
    assert kkk["L_int"] == pytest.approx(40 * (0.5 + 0.1) / 2) and kkk["filled_L_int"] == "neighbour_ratio"
    gbr = out.loc[("GBR", 2014)]
    assert gbr["total"] == pytest.approx(90)  # territorial total kept
    assert gbr["L_int"] == pytest.approx(18) and gbr["L_dom"] == pytest.approx(72)  # L_int/T 0.2 held
    assert gbr["filled_L_int"] == "own_trend"
    irl = out.loc[("IRL", 2015)]
    assert irl["L_int_foreign"] == pytest.approx(30) and irl["L_int"] == pytest.approx(40)
    assert irl["filled_L_int"] == "own_trend" and out.loc[("IRL", 2010), "filled_L_int"] == "observed"
    # Components add up everywhere.
    assert np.allclose(out["L_long"] + out["L_short"], out["L_dom"])
    assert np.allclose(out["L_long"], out["alpha_E"] * out["L_dom"])
    assert np.allclose(out["total"], out["L_dom"] + out["L_int"])
    assert not out[["L_dom", "L_int", "alpha_E"]].isna().any().any()
