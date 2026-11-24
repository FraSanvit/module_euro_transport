"""Unit tests for the heavy-duty level chain (MULTIYEAR_PLAN.md §6.3-§6.8)."""

import logging

import numpy as np
import pandas as pd
import pytest
from _heavy_duty import (
    alpha_cap,
    annual_levels,
    apply_exclusions,
    cabotage_load_factor,
    cabotage_vkm,
    calibrate,
    domestic_level,
    international_index,
    override_split,
    short_trip_share,
    survey_activity,
    zero_missing_empty,
)

T0 = 2019


def tott_rows(geo, year, **values):
    """road_go_ta_tott rows; keys like LNAT_vkm, LINTL_CAB_tkm (million)."""
    rows = []
    for key, value in values.items():
        oper, unit = key.rsplit("_", 1)
        rows.append({"geo": geo, "tra_oper": oper, "unit": f"MIO_{unit.upper()}", "year": year, "value": value})
    return rows


def survey_table():
    rows = (
        tott_rows("AA", T0, LNAT_vkm=100, ENAT_vkm=20, LINTL_vkm=50, EINTL_vkm=10, LINTL_CAB_vkm=2, LINTL_CAB_tkm=30)
        + tott_rows("BB", T0, LNAT_vkm=200, ENAT_vkm=40, LINTL_vkm=10, EINTL_vkm=0)
        + tott_rows("EU27_2020", T0, LINTL_CAB_vkm=10, LINTL_CAB_tkm=150)
    )  # fmt: skip
    return survey_activity(pd.DataFrame(rows))


def test_survey_activity_units():
    s = survey_table()
    assert s.loc[("AA", T0), "LNAT"] == 100e6
    assert s.loc[("AA", T0), "CAB_OUT_TKM"] == 30e6


def test_zero_missing_empty_lower_bound(caplog):
    rows = tott_rows("IT", T0, LNAT_vkm=100, LINTL_vkm=10) + tott_rows("AA", T0, LNAT_vkm=50, ENAT_vkm=5)
    with caplog.at_level(logging.WARNING):
        s = zero_missing_empty(survey_activity(pd.DataFrame(rows)))
    assert s.loc[("IT", T0), "ENAT"] == 0
    assert s.loc[("IT", T0), "EINTL"] == 0
    assert s.loc[("IT", T0), "empty_zero_nat"]
    assert s.loc[("IT", T0), "empty_zero_intl"]
    assert not s.loc[("AA", T0), "empty_zero_nat"]
    assert s.loc[("AA", T0), "ENAT"] == 5e6
    assert "IT: no ENAT reported" in caplog.text
    # Domestic: flagged as a lower bound. Own international: only where the reference
    # year had empty km (AA in 2020), not where r_o was calibrated laden-only (TT).
    stats, network, totals, toll = chain_inputs()
    stats["empty_zero_nat"] = stats["geo"] == "AA"
    stats["empty_zero_intl"] = ((stats["geo"] == "AA") & (stats["year"] == 2020)) | (stats["geo"] == "TT")
    cal = calibrate(stats, network, totals, toll, T0, 1.0, 0.8, 0.9, ["BB"], ["DE"])
    lv = annual_levels(stats, cal, totals, toll, [T0, 2020], 0.8, 0.9, ["BB"], ["DE"]).set_index(["geo", "year"])
    assert "empty_km_zero_lower_bound" in lv.loc[("AA", T0), "flags"]
    assert "empty_km_zero_lower_bound" not in lv.loc[("BB", T0), "flags"]
    assert "own_int_empty_km_zero_lower_bound" in lv.loc[("AA", 2020), "flags"]
    assert "own_int_empty_km_zero_lower_bound" not in lv.loc[("AA", T0), "flags"]
    assert "own_int_empty_km_zero_lower_bound" not in lv.loc[("TT", 2020), "flags"]


def test_short_trip_share():
    dc = pd.DataFrame({"geo": ["AA"], "distance": ["KM_LT50"], "year": [T0], "value": [25.0]})
    assert short_trip_share(dc, survey_table()).loc[("AA", T0)] == pytest.approx(0.75)


def test_cabotage_load_factor_own_then_eu():
    lf = cabotage_load_factor(survey_table()).set_index("geo")
    assert lf.loc["AA", "load_factor"] == pytest.approx(15.0)  # 30 / 2
    assert lf.loc["BB", "load_factor"] == pytest.approx(15.0)  # EU27 150 / 10
    assert lf.loc["BB", "lf_source"] == "EU27"


def test_cabotage_vkm_host_empty_share_and_zero_hosts():
    # AA hauliers do 300 thousand tkm of cabotage in BB: 300e3 / 15 = 20,000 laden vkm.
    hac = pd.DataFrame(
        {"geo": ["AA", "BB", "AA"], "c_cabot": ["BB", "BB", "CC"], "year": [T0] * 3, "value": [300.0, 99.0, 150.0]}
    )  # fmt: skip
    cab = cabotage_vkm(hac, survey_table(), hosts_zero=["CC"]).set_index("geo")
    assert cab.loc["BB", "CAB_laden"] == pytest.approx(20_000)
    assert cab.loc["BB", "CAB"] == pytest.approx(20_000 * 1.2)  # BB empty share 40/200
    assert cab.loc["CC", "CAB"] == 0  # no cabotage rights
    fixed = cabotage_vkm(hac, survey_table(), empty_share=0).set_index("geo")
    assert fixed.loc["BB", "CAB"] == pytest.approx(20_000)


def test_domestic_level():
    stats = pd.DataFrame({"geo": ["AA", "BB", "CC"], "LNAT": [100.0, 200, 50], "ENAT": [20.0, np.nan, 5], "CAB": [3.0, 1, np.nan]})
    d = domestic_level(stats, hosts_zero=["AA"]).set_index("geo")["D"]
    assert d["AA"] == 120  # cabotage zeroed
    assert np.isnan(d["BB"])  # no empty km: left missing
    assert d["CC"] == 55  # no cabotage data: 0
    assert domestic_level(stats, include_cabotage=False).set_index("geo")["D"]["CC"] == 55


def ia_rows(geo, load, unload, year, value):
    return {"geo": geo, "c_load": load, "c_unload": unload, "year": year, "value": value}


def test_international_index_foreign_own_transit_and_coverage(caplog):
    ia = pd.DataFrame(
        [
            ia_rows("AA", "AA", "BB", T0, 10),  # own AA; foreign for BB
            ia_rows("BB", "CC", "AA", T0, 4),  # foreign for AA (unloaded in AA)
            ia_rows("UK", "AA", "CC", T0, 6),  # foreign for AA
            ia_rows("AA", "AA", "AA", T0, 99),  # national: ignored
            ia_rows("BB", "CC", "AA", 2020, 4),
            ia_rows("AA", "AA", "BB", 2020, 10),
        ]
    )
    tra = pd.DataFrame({"geo": ["BB"], "c_trans": ["AA"], "year": [T0], "value": [5.0]})
    with caplog.at_level(logging.INFO):
        idx = international_index(ia, tra, T0).set_index(["geo", "year"])
    assert idx.loc[("AA", T0), "I_f"] == pytest.approx((4 + 6 + 5) * 1e3)
    assert idx.loc[("AA", T0), "J_o"] == pytest.approx(10e3)
    # 2020: UK does not report; its 6 of 15 kt at T0 -> m = 0.4.
    assert idx.loc[("AA", 2020), "coverage_m"] == pytest.approx(0.4)
    assert idx.loc[("AA", 2020), "I_f_corrected"] == pytest.approx(4e3 / 0.6)
    assert "UK" in caplog.text
    # With a 0.3 limit the 2020 index is left missing (40% from absent hauliers).
    strict = international_index(ia, tra, T0, max_coverage_gap=0.3).set_index(["geo", "year"])
    assert np.isnan(strict.loc[("AA", 2020), "I_f_corrected"])
    assert strict.loc[("AA", T0), "I_f_corrected"] == pytest.approx(15e3)


def test_alpha_cap():
    cap = alpha_cap(pd.Series([0.5, 0.95, np.nan]), f_net=0.9, alpha_max=0.8)
    assert cap.tolist() == pytest.approx([0.45, 0.8, 0.8])


def test_override_split():
    l_dom, l_int = override_split(np.array([100.0, 100, 100]), np.array([30.0, 80, 120]), np.array([50.0, 50, np.nan]))
    assert l_dom.tolist() == [70, 50, 0]  # T - L_int; floor LNAT+ENAT; capped at T
    assert l_int.tolist() == [30, 50, 100]


def test_apply_exclusions(caplog):
    stats = pd.DataFrame({"geo": ["ME", "ME"], "year": [2018, 2019], "LNAT": [1.0, 2], "ENAT": [1.0, 2],
                          "alpha_E": [0.5, 0.5], "LINTL": [1.0, 1], "EINTL": [1.0, 1], "I_o": [2.0, 2], "J_o": [1.0, 1]})  # fmt: skip
    with caplog.at_level(logging.WARNING):
        out = apply_exclusions(stats, {"MNE": {"domestic": "all", "own_international": [2019]}}, {"MNE": "ME"})
    assert out["LNAT"].isna().all()
    assert out["I_o"].tolist()[0] == 2 and np.isnan(out["I_o"].tolist()[1])
    assert "MNE domestic" in caplog.text
    # Countries not processed are skipped.
    assert len(apply_exclusions(stats, {"MKD": {"domestic": "all"}}, {"MNE": "ME"})) == 2


def chain_inputs():
    """Two years for AA (statistics), BB (island), TT (territorial total) and DE (toll)."""
    base = {"LNAT": 80.0, "ENAT": 20.0, "CAB": 0.0, "D": 100.0, "I_f": 50.0, "coverage_m": 0.0,
            "I_f_corrected": 50.0, "J_o": 50.0, "I_o": 40.0, "alpha_E": 0.5}  # fmt: skip
    rows = []
    for geo in ["AA", "BB", "TT", "DE"]:
        for year, scale in [(T0, 1.0), (2020, 1.1)]:
            row = {"geo": geo, "year": year, **base}
            row["I_f_corrected"] = 50.0 * scale
            row["I_o"] = 40.0 * scale
            row["D"] = 100.0 * scale
            row["CAB"] = 10.0 if geo == "DE" else 0.0
            rows.append(row)
    stats = pd.DataFrame(rows)
    network = pd.DataFrame({"geo": ["AA", "BB", "TT", "DE"], "etis_int": [60.0, 60, 60, 60], "etis_dom": [30.0, 30, 90, 30]})
    totals = pd.DataFrame({"geo": ["TT", "TT"], "year": [T0, 2020], "T": [150.0, 170.0]})
    toll = pd.DataFrame({"geo": ["DE", "DE"], "year": [T0, 2020], "toll_foreign": [40.0, 50.0]})
    return stats, network, totals, toll


def run_chain(k=1.0):
    stats, network, totals, toll = chain_inputs()
    cal = calibrate(stats, network, totals, toll, T0, k, 0.8, 0.9, ["BB"], ["DE"])
    levels = annual_levels(stats, cal, totals, toll, [T0, 2020], 0.8, 0.9, ["BB"], ["DE"])
    return cal.set_index("geo"), levels.set_index(["geo", "year"])


def test_calibration():
    cal, _ = run_chain()
    aa = cal.loc["AA"]
    assert aa["w0"] == pytest.approx(0.5)  # I_f 50 vs J_o 50
    assert aa["r_f"] == pytest.approx(30 / 50)  # L^f0 = 0.5 x 60
    assert aa["r_o"] == pytest.approx(30 / 40)
    assert aa["s"] == pytest.approx(min(30 / 100, 0.45))
    assert cal.loc["BB", "L_int0"] == 0  # island
    # TT: T 150, ETIS L_int 60 -> T - L_int = 90 is below LNAT + ENAT = 100, so
    # L_dom = 100 and L_int is cut to 50 (§6.8); s = min(90/100, 0.45) capped.
    assert cal.loc["TT", "L_dom0"] == pytest.approx(100)
    assert cal.loc["TT", "L_int0"] == pytest.approx(50)
    assert "int_adjusted_t0" in cal.loc["TT", "calibration_flags"]
    assert cal.loc["TT", "s"] == pytest.approx(0.45)
    assert "long_capped_t0" in cal.loc["TT", "calibration_flags"]
    # DE: gamma = L^f0 / (toll 40 - cabotage 10) = 30 / 30.
    assert cal.loc["DE", "gamma"] == pytest.approx(1.0)


def test_annual_levels_chain():
    _, lv = run_chain()
    aa = lv.loc[("AA", 2020)]
    assert aa["L_int_foreign"] == pytest.approx(0.6 * 55)
    assert aa["L_int_own"] == pytest.approx(0.75 * 44)
    assert aa["L_long"] == pytest.approx(0.3 * 110)
    assert aa["L_short"] == pytest.approx(110 - 33)
    assert aa["total"] == pytest.approx(aa["L_int"] + 110)
    assert lv.loc[("AA", T0), "total"] == pytest.approx(60 + 100)  # reproduces the calibration
    assert lv.loc[("BB", 2020), "L_int"] == 0
    de = lv.loc[("DE", 2020)]
    assert de["L_int_foreign"] == pytest.approx(1.0 * (50 - 10))  # toll path
    assert de["level_source"] == "survey + toll"


def test_annual_levels_territorial_override():
    _, lv = run_chain()
    tt = lv.loc[("TT", 2020)]
    assert tt["total"] == 170
    assert tt["L_int"] + tt["L_dom"] == pytest.approx(170)
    assert tt["L_long"] + tt["L_short"] == pytest.approx(tt["L_dom"])
    assert tt["level_source"] == "territorial total"


def test_missing_inputs_stay_missing():
    stats, network, totals, toll = chain_inputs()
    stats.loc[stats["geo"] == "AA", ["D", "ENAT"]] = np.nan
    cal = calibrate(stats, network, totals, toll, T0, 1.0, 0.8, 0.9, ["BB"], ["DE"])
    lv = annual_levels(stats, cal, totals, toll, [T0, 2020], 0.8, 0.9, ["BB"], ["DE"]).set_index(["geo", "year"])
    assert np.isnan(lv.loc[("AA", 2020), "total"])
    assert not np.isnan(lv.loc[("AA", 2020), "L_int"])  # international still computed
