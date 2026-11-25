"""Check the heavy-duty national levels (before gap filling) produced in the integration test."""

import numpy as np
import pandas as pd

levels = pd.read_parquet(snakemake.input.levels)  # noqa: F821
calibration = pd.read_parquet(snakemake.input.calibration).set_index("country_id")  # noqa: F821
manifest = pd.read_csv(snakemake.input.manifest)  # noqa: F821
countries = snakemake.params.countries  # noqa: F821

assert set(countries) <= set(levels["country_id"])
assert not levels.duplicated(["country_id", "year"]).any()

# Components add up wherever they exist.
ok = levels.dropna(subset=["L_int", "L_dom", "L_long", "L_short", "total"])
assert np.allclose(ok["L_int"] + ok["L_dom"], ok["total"], rtol=1e-9)
assert np.allclose(ok["L_long"] + ok["L_short"], ok["L_dom"], rtol=1e-9)
assert np.allclose(ok["L_long"], ok["s_long"] * ok["L_dom"], rtol=1e-9)

y = levels[levels["year"] == 2019].set_index("country_id")
# DEU: territorial reference 48.9-51.8 bn vkm; international 16-22 bn (MULTIYEAR_PLAN.md E6).
assert 45e9 < y.loc["DEU", "total"] < 56e9
assert 16e9 < y.loc["DEU", "L_int"] < 23e9
assert y.loc["DEU", "source"] == "survey + toll"
assert np.isfinite(calibration.loc["DEU", "gamma"])
# Territorial totals fix the total (GBR: DfT + Northern Ireland; CHE: BFS).
for country in ["GBR", "CHE"]:
    assert y.loc[country, "total"] == y.loc[country, "T"]
    assert y.loc[country, "source"].startswith("territorial total")
assert 27e9 < y.loc["GBR", "total"] < 30e9
# Island: no international traffic.
assert y.loc["ISL", "L_int"] == 0
# No usable domestic statistic: left missing for the gap-filling step.
for country in ["BIH", "MKD", "XKX", "ISL"]:
    assert np.isnan(y.loc[country, "total"]), country
    assert "no_domestic_statistic" in y.loc[country, "flags"]
# Survey countries: plausible vkm per year.
for country in ["BEL", "NLD", "LUX", "NOR"]:
    assert 0.3e9 < y.loc[country, "total"] < 10e9, country

# After gap filling: every country and year complete, components consistent (§6.9).
filled = pd.read_parquet(snakemake.input.filled)  # noqa: F821
assert set(countries) <= set(filled["country_id"])
assert not filled[["L_int", "L_dom", "L_long", "L_short", "total", "alpha_E"]].isna().any().any()
assert np.allclose(filled["L_int"] + filled["L_dom"], filled["total"], rtol=1e-9)
assert np.allclose(filled["L_long"], filled["alpha_E"] * filled["L_dom"], rtol=1e-9)
f19 = filled[filled["year"] == 2019].set_index("country_id")
for country in ["BIH", "MKD", "XKX", "ISL"]:
    assert f19.loc[country, "filled_L_dom"].startswith("neighbours_gdp"), country
    assert 0.02e9 < f19.loc[country, "total"] < 2e9, country
assert f19.loc["XKX", "filled_L_dom"] == "neighbours_gdp_eur"
observed = filled.merge(levels, on=["country_id", "year"], suffixes=("", "_raw")).dropna(subset=["total_raw"])
assert np.allclose(observed["total"], observed["total_raw"], rtol=1e-9)  # observed totals unchanged

# Every download is in the manifest with a checksum; ETISplus matches Mendeley's.
assert manifest["sha256"].str.len().eq(64).all()
etis = manifest[manifest["path"].str.contains("/etisplus/")]
assert len(etis) == 4

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
