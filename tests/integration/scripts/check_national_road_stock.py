"""Check the combined national road vehicle stock produced in the integration test."""

import pandas as pd

stock = pd.read_parquet(snakemake.input.stock)  # noqa: F821
countries = snakemake.params.countries  # noqa: F821
# One stock source per country.
SOURCES = {
    "BEL": "JRC-IDEES-2021 v1",
    "NLD": "JRC-IDEES-2021 v1",
    "LUX": "JRC-IDEES-2021 v1",
    "NOR": "Eurostat road_eqs",
    "CHE": "BFS cc-d-11.03.02.01.03/je-d-11.03.02.01.04",
    "GBR": "DfT VEH0101",
}

assert set(stock["country_id"]) == set(SOURCES) & set(countries)
assert set(stock["unit"]) == {"vehicles"}
assert not stock.duplicated(["country_id", "year", "vehicle_category"]).any()
for country, source in SOURCES.items():
    assert set(stock.loc[stock["country_id"] == country, "source"]) == {source}
jrc = stock[stock["source"] == "JRC-IDEES-2021 v1"]
assert set(jrc["year"]) == set(range(2000, 2022))

# Plausibility: more passenger cars than heavy-duty vehicles wherever both exist.
wide = stock.pivot_table(
    index=["country_id", "year"], columns="vehicle_category", values="stock"
).dropna(subset=["passenger_car", "heavy_duty"])
assert (wide["passenger_car"] > wide["heavy_duty"]).all()

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
