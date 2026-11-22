"""Check the combined national road vehicle-km produced in the integration test."""

import pandas as pd

vkm = pd.read_parquet(snakemake.input.vkm)  # noqa: F821
countries = snakemake.params.countries  # noqa: F821
# Neighbours used for gap filling are processed too (JRC-IDEES and Eurostat).
NEIGHBOURS = {"BGR", "GRC", "HRV", "HUN", "ROU", "SVN", "DNK", "NOR", "SWE"}
assert set(vkm["country_id"]) - set(countries) == NEIGHBOURS - set(countries)
vkm = vkm[vkm["country_id"].isin(countries)]
EU = {"BEL", "NLD", "LUX"}
NO_VKM = {"ALB", "BIH", "MNE", "SRB", "XKX", "ISL"}  # no reliable source

assert sorted(vkm["country_id"].unique()) == sorted(set(countries) - NO_VKM)
assert (vkm.groupby("country_id")["vkm"].sum() > 0).all()

jrc = vkm[vkm["country_id"].isin(EU)]
assert set(jrc["source"]) == {"JRC-IDEES-2021 v1"}
assert set(jrc["year"]) == set(range(2000, 2022))
assert len(jrc) == len(EU) * 22 * 5

eurostat = vkm[vkm["country_id"].isin({"GBR", "NOR"})]
assert set(eurostat["source"]) == {"Eurostat road_tf_vehmov/road_tf_road"}
assert set(eurostat["principle"]) == {"territorial"}
assert eurostat["year"].min() >= 2013
# GB heavy goods vehicles 2019: about 28 bn vkm (DfT road traffic estimates).
gbr_hdv = eurostat.query(
    "country_id == 'GBR' and vehicle_category == 'heavy_duty' and year == 2019"
)["vkm"]
assert 25e9 < gbr_hdv.item() < 31e9

bfs = vkm[vkm["country_id"] == "CHE"]
assert set(bfs["source"]) == {"BFS je-d-11.04.01.01/je-d-11.05.01.01"}
assert set(bfs["principle"]) == {"territorial"}
assert bfs["year"].min() == 1990
assert set(bfs["vehicle_category"]) == {
    "heavy_duty",
    "light_duty",
    "passenger_car",
    "motorcycle",
    "bus",
}
# Swiss heavy goods vehicles 2019: 2,226 Mvkm (BFS je-d-11.05.01.01).
che_hdv = bfs.query("vehicle_category == 'heavy_duty' and year == 2019")["vkm"]
assert 2.0e9 < che_hdv.item() < 2.5e9

# MKD: heavy-duty from MAKSTAT (nationality), cars and buses from Eurostat.
mkd = vkm[vkm["country_id"] == "MKD"].set_index("vehicle_category")
assert set(mkd.index) == {"heavy_duty", "passenger_car", "bus"}
assert set(mkd.loc["heavy_duty", "source"]) == {"MAKSTAT 560_Trans_MK_T39"}
assert set(mkd.loc["heavy_duty", "principle"]) == {"nationality"}
assert mkd.loc["heavy_duty", "year"].min() == 2004
assert set(mkd.loc[["passenger_car", "bus"], "source"]) == {
    "Eurostat road_tf_vehmov/road_tf_road"
}
# North Macedonian goods vehicles 2019: 835 Mvkm (MAKSTAT 560_Trans_MK_T39).
mkd_hdv = mkd.reset_index().query("vehicle_category == 'heavy_duty' and year == 2019")
assert mkd_hdv["vkm"].item() == 835e6

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
