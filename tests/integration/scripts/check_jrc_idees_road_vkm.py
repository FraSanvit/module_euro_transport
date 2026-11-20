"""Check the combined JRC-IDEES road vehicle-km produced in the integration test."""

import pandas as pd

vkm = pd.read_parquet(snakemake.input.vkm)  # noqa: F821
countries = snakemake.params.countries  # noqa: F821

assert sorted(vkm["country_id"].unique()) == sorted(countries)
assert set(vkm["year"]) == set(range(2000, 2022))
assert vkm["vehicle_category"].nunique() == 5
assert len(vkm) == len(countries) * 22 * 5
assert (vkm.groupby("country_id")["vkm"].sum() > 0).all()

with open(snakemake.output[0], "w") as f:  # noqa: F821
    f.write("ok\n")
