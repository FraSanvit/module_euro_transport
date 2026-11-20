"""Check the combined national road vehicle-km produced in the integration test."""

import pandas as pd

vkm = pd.read_parquet(snakemake.input.vkm)  # noqa: F821
countries = snakemake.params.countries  # noqa: F821
EU = {"BEL", "NLD", "LUX"}

assert sorted(vkm["country_id"].unique()) == sorted(countries)
assert (vkm.groupby("country_id")["vkm"].sum() > 0).all()

jrc = vkm[vkm["country_id"].isin(EU)]
assert set(jrc["source"]) == {"JRC-IDEES-2021 v1"}
assert set(jrc["year"]) == set(range(2000, 2022))
assert len(jrc) == len(EU) * 22 * 5

eurostat = vkm[~vkm["country_id"].isin(EU)]
assert set(eurostat["source"]) == {"Eurostat road_tf_vehmov/road_tf_road"}
assert set(eurostat["principle"]) == {"territorial"}
assert eurostat["year"].min() >= 2013
# GB heavy goods vehicles 2019: about 28 bn vkm (DfT road traffic estimates).
gbr_hdv = eurostat.query(
    "country_id == 'GBR' and vehicle_category == 'heavy_duty' and year == 2019"
)["vkm"]
assert 25e9 < gbr_hdv.item() < 31e9

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
