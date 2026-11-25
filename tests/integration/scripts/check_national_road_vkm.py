"""Check the combined national road vehicle-km produced in the integration test."""

import pandas as pd

vkm = pd.read_parquet(snakemake.input.vkm)  # noqa: F821
countries = snakemake.params.countries  # noqa: F821
# Neighbours used for gap filling are processed too (JRC-IDEES and Eurostat).
NEIGHBOURS = {"BGR", "GRC", "HRV", "HUN", "ROU", "DNK", "FIN", "NOR", "SWE"}
assert set(vkm["country_id"]) - set(countries) == NEIGHBOURS - set(countries)
vkm = vkm[vkm["country_id"].isin(countries)]
EU = {"BEL", "NLD", "LUX"}
HDV_ONLY = {"BIH", "XKX", "ISL"}  # no reliable source for the other categories
HDV_YEARS = {2018, 2019, 2020, 2021}  # heavy_duty years of the test config

assert sorted(vkm["country_id"].unique()) == sorted(countries)
assert (vkm.groupby("country_id")["vkm"].sum() > 0).all()

# heavy_duty: territorial totals of the heavy-duty level chain, every configured country.
hdv = vkm[vkm["vehicle_category"] == "heavy_duty"]
assert hdv["source"].str.startswith("HDV level chain").all()
assert set(hdv["principle"]) == {"territorial"}
assert set(hdv["year"]) == HDV_YEARS
assert len(hdv) == len(countries) * len(HDV_YEARS)
assert set(vkm[vkm["country_id"].isin(HDV_ONLY)]["vehicle_category"]) == {"heavy_duty"}
for country in HDV_ONLY | {"MKD"}:
    assert "(filled:" in hdv.query("country_id == @country and year == 2019")["source"].item(), country
other = vkm[vkm["vehicle_category"] != "heavy_duty"]

jrc = other[other["country_id"].isin(EU)]
assert set(jrc["source"]) == {"JRC-IDEES-2021 v1"}
assert set(jrc["year"]) == set(range(2000, 2022))
assert len(jrc) == len(EU) * 22 * 4

eurostat = other[other["country_id"].isin({"GBR", "NOR"})]
assert set(eurostat["source"]) == {"Eurostat road_tf_vehmov/road_tf_road"}
assert set(eurostat["principle"]) == {"territorial"}
assert eurostat["year"].min() >= 2013
# GB heavy goods vehicles 2019: about 28 bn vkm (DfT road traffic estimates; level-chain override).
gbr_hdv = hdv.query("country_id == 'GBR' and year == 2019")["vkm"]
assert 25e9 < gbr_hdv.item() < 31e9

bfs = other[other["country_id"] == "CHE"]
assert set(bfs["source"]) == {"BFS je-d-11.04.01.01/je-d-11.05.01.01"}
assert set(bfs["principle"]) == {"territorial"}
assert bfs["year"].min() == 1990
assert set(bfs["vehicle_category"]) == {"light_duty", "passenger_car", "motorcycle", "bus"}
# Swiss heavy goods vehicles 2019: 2,226 Mvkm (BFS je-d-11.05.01.01; level-chain override).
che_hdv = hdv.query("country_id == 'CHE' and year == 2019")["vkm"]
assert 2.0e9 < che_hdv.item() < 2.5e9

# MKD: heavy-duty from the level chain (filled: no Eurostat survey), cars and buses from Eurostat.
mkd = vkm[vkm["country_id"] == "MKD"].set_index("vehicle_category")
assert set(mkd.index) == {"heavy_duty", "passenger_car", "bus"}
assert set(mkd.loc[["passenger_car", "bus"], "source"]) == {
    "Eurostat road_tf_vehmov/road_tf_road"
}
mkd_hdv = hdv.query("country_id == 'MKD' and year == 2019")["vkm"]
assert 0.1e9 < mkd_hdv.item() < 1e9

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
