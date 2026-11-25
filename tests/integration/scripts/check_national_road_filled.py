"""Check the gap-filled national tables produced in the integration test."""

import pandas as pd

KEYS = ["country_id", "year", "vehicle_category"]

countries = snakemake.params.countries  # noqa: F821

for metric in ["vkm", "stock"]:
    observed = pd.read_parquet(snakemake.input[metric])  # noqa: F821
    observed = observed[observed["country_id"].isin(countries)]
    filled = pd.read_parquet(snakemake.input[f"{metric}_filled"])  # noqa: F821
    assert set(filled["country_id"]) <= set(countries)  # neighbours are not written
    assert not filled.duplicated(KEYS).any()
    # Observed values are kept unchanged.
    kept = filled[filled["filled"] == "observed"].drop(columns="filled")
    pd.testing.assert_frame_equal(
        kept.sort_values(KEYS, ignore_index=True),
        observed.sort_values(KEYS, ignore_index=True),
    )
    # Filled values lie inside the observed range of their series.
    added = filled[filled["filled"] == "interpolated"]
    bounds = observed.groupby(["country_id", "vehicle_category"])["year"].agg(["min", "max"])
    joined = added.join(bounds, on=["country_id", "vehicle_category"])
    assert ((joined["year"] > joined["min"]) & (joined["year"] < joined["max"])).all()

stock = pd.read_parquet(snakemake.input.stock_filled).set_index(KEYS)  # noqa: F821
vkm = pd.read_parquet(snakemake.input.vkm_filled).set_index(KEYS)  # noqa: F821
# NOR car stock 2001-2003 and XKX goods vehicles 2018-2019 (excluded) are interpolated.
for key in [("NOR", 2002, "passenger_car"), ("XKX", 2018, "heavy_duty"), ("BIH", 2012, "passenger_car")]:
    assert stock.loc[key, "filled"] == "interpolated", key
# GBR vehicle-km 2018 is interpolated for all categories but heavy_duty (complete from the
# heavy-duty level chain).
gbr = vkm.xs(("GBR", 2018), level=["country_id", "year"])
assert len(gbr) == 5 and gbr.loc["heavy_duty", "filled"] == "observed"
assert set(gbr.drop(index="heavy_duty")["filled"]) == {"interpolated"}

# Categories missing entirely are filled from neighbour ratios to cars, for the
# years with own car stock: MKD light, heavy, motorcycle; XKX motorcycle.
ratio = stock[stock["filled"] == "neighbour_ratio"].reset_index()
assert set(map(tuple, ratio[["country_id", "vehicle_category"]].drop_duplicates().values)) == {
    ("MKD", "light_duty"), ("MKD", "heavy_duty"), ("MKD", "motorcycle"), ("XKX", "motorcycle"),
}
mkd = stock.xs("MKD", level="country_id")["stock"].unstack("vehicle_category")
assert mkd.notna().all().all() and len(mkd) == 14  # 2012-2025, all five categories
assert (mkd["heavy_duty"] < mkd["light_duty"]).all() and (mkd["light_duty"] < mkd["passenger_car"]).all()

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
