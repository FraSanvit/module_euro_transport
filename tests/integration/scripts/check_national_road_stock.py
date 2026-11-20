"""Check the combined national road vehicle stock produced in the integration test."""

import pandas as pd

stock = pd.read_parquet(snakemake.input.stock)  # noqa: F821
EU = {"BEL", "NLD", "LUX"}

# Only the EU countries have a stock source so far (JRC-IDEES).
assert set(stock["country_id"]) == EU
assert set(stock["source"]) == {"JRC-IDEES-2021 v1"}
assert set(stock["unit"]) == {"vehicles"}
assert set(stock["year"]) == set(range(2000, 2022))
assert len(stock) == len(EU) * 22 * 5
# Plausibility: more passenger cars than heavy-duty vehicles everywhere.
wide = stock.pivot_table(
    index=["country_id", "year"], columns="vehicle_category", values="stock"
)
assert (wide["passenger_car"] > wide["heavy_duty"]).all()

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
