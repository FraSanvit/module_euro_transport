"""Check the gap-filled national tables produced in the integration test."""

import pandas as pd

KEYS = ["country_id", "year", "vehicle_category"]

for metric in ["vkm", "stock"]:
    observed = pd.read_parquet(snakemake.input[metric])  # noqa: F821
    filled = pd.read_parquet(snakemake.input[f"{metric}_filled"])  # noqa: F821
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
# GBR vehicle-km 2018 is interpolated for all categories.
gbr = vkm.xs(("GBR", 2018), level=["country_id", "year"])
assert len(gbr) == 5 and set(gbr["filled"]) == {"interpolated"}

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
