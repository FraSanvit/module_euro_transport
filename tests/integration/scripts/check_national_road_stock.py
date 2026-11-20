"""Check the combined national road vehicle stock produced in the integration test."""

import pandas as pd

stock = pd.read_parquet(snakemake.input.stock)  # noqa: F821
countries = snakemake.params.countries  # noqa: F821
EU = {"BEL", "NLD", "LUX"}
JRC, EUROSTAT = "JRC-IDEES-2021 v1", "Eurostat road_eqs"

assert set(stock["country_id"]) == set(countries)
assert set(stock["unit"]) == {"vehicles"}
assert not stock.duplicated(["country_id", "year", "vehicle_category"]).any()

# Priority: JRC-IDEES wherever it has a value (EU, 2000-2021), Eurostat otherwise.
eu = stock[stock["country_id"].isin(EU)]
assert set(eu.loc[eu["year"].between(2000, 2021), "source"]) == {JRC}
assert set(eu.loc[eu["year"] > 2021, "source"]) <= {EUROSTAT}
assert set(stock.loc[~stock["country_id"].isin(EU), "source"]) == {EUROSTAT}

# Plausibility: more passenger cars than heavy-duty vehicles wherever both exist.
wide = stock.pivot_table(
    index=["country_id", "year"], columns="vehicle_category", values="stock"
).dropna(subset=["passenger_car", "heavy_duty"])
assert (wide["passenger_car"] > wide["heavy_duty"]).all()

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
