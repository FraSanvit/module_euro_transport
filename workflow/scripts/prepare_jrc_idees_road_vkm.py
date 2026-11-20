"""Extract annual road vehicle-km per vehicle category from a JRC-IDEES 2021 country zip.

Reads sheet `TrRoad_act` of `JRC-IDEES-2021_Transport_{CC}.xlsx` and selects rows by
their JRC code (last column), not by position.
"""

import sys

import pandas as pd
from _jrc_idees import (  # noqa: F401 (re-exported for tests)
    read_road_activity,
    select_categories,
    tidy_activity_sheet,
)
from _schemas import NationalVkmSchema

PREFIX = "VehicleKm.Mkm"
MKM_TO_KM = 1e6


def extract_vkm(
    activity: pd.DataFrame, jrc_code: str, country_id: str, source: str
) -> pd.DataFrame:
    """Select the vehicle-km rows per category and return them in long format.

    Raises if a category code is missing or if the categories do not add up to
    the total road vehicle-km.
    """
    long = select_categories(activity, PREFIX, jrc_code, country_id)
    long["vkm"] = long.pop("value") * MKM_TO_KM
    long["country_id"] = country_id
    long["unit"] = "vkm"
    long["source"] = source
    long["principle"] = "unspecified"
    long = long[list(NationalVkmSchema.columns)]
    long = long.sort_values(["year", "vehicle_category"], ignore_index=True)
    return NationalVkmSchema.validate(long)


def main(zip_path, jrc_code, country_id, source, output_path):
    """Extract one country and write it as parquet."""
    activity = read_road_activity(zip_path, jrc_code)
    vkm = extract_vkm(activity, jrc_code, country_id, source)
    vkm.to_parquet(output_path, index=False)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
    main(
        zip_path=snakemake.input.zip,  # noqa: F821
        jrc_code=snakemake.params.jrc_code,  # noqa: F821
        country_id=snakemake.wildcards.country,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        output_path=snakemake.output.vkm,  # noqa: F821
    )
