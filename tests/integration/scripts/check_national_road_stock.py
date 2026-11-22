"""Check the combined national road vehicle stock produced in the integration test."""

import pandas as pd

stock = pd.read_parquet(snakemake.input.stock)  # noqa: F821
countries = snakemake.params.countries  # noqa: F821
# Neighbours used for gap filling are processed too (JRC-IDEES and Eurostat).
NEIGHBOURS = {"BGR", "HRV", "HUN", "ROU", "SVN", "DNK", "NOR", "SWE"}
assert set(stock["country_id"]) - set(countries) == NEIGHBOURS - set(countries)
stock = stock[stock["country_id"].isin(countries)]
# One stock source per country.
SOURCES = {
    "BEL": "JRC-IDEES-2021 v1",
    "NLD": "JRC-IDEES-2021 v1",
    "LUX": "JRC-IDEES-2021 v1",
    "NOR": "Eurostat road_eqs",
    "CHE": "BFS cc-d-11.03.02.01.03/je-d-11.03.02.01.04",
    "GBR": "DfT VEH0101",
    "MKD": "MAKSTAT 150_Trans_regi_veh_by_type_mun",
    "BIH": "Eurostat road_eqs",
    "ISL": "Statistics Iceland SAM03105",
    "XKX": "ASK tr05",
}

assert set(stock["country_id"]) == set(SOURCES) & set(countries)
assert set(stock["unit"]) == {"vehicles"}
assert not stock.duplicated(["country_id", "year", "vehicle_category"]).any()
for country, source in SOURCES.items():
    assert set(stock.loc[stock["country_id"] == country, "source"]) == {source}
jrc = stock[stock["source"] == "JRC-IDEES-2021 v1"]
assert set(jrc["year"]) == set(range(2000, 2022))
# MKD: goods vehicles are not split by weight and motor-cycles exclude mopeds.
mkd = stock[stock["country_id"] == "MKD"]
assert set(mkd["vehicle_category"]) == {"passenger_car", "bus"}
assert mkd["year"].min() == 2012
# BIH: all five categories; ISL: all five from 1999; XKX: no motorcycles, goods
# vehicles 2018-2019 excluded (swapped rows in the source).
assert stock.query("country_id == 'BIH'")["vehicle_category"].nunique() == 5
isl = stock[stock["country_id"] == "ISL"]
assert isl["vehicle_category"].nunique() == 5 and isl["year"].min() == 1999
xkx = stock[stock["country_id"] == "XKX"]
assert "motorcycle" not in set(xkx["vehicle_category"])
assert not xkx.query("vehicle_category == 'heavy_duty' and year in [2018, 2019]").size

# Plausibility: more passenger cars than heavy-duty vehicles wherever both exist.
wide = stock.pivot_table(
    index=["country_id", "year"], columns="vehicle_category", values="stock"
).dropna(subset=["passenger_car", "heavy_duty"])
assert (wide["passenger_car"] > wide["heavy_duty"]).all()

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
