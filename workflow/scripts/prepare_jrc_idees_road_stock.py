"""Extract the road vehicle stock per vehicle category from a JRC-IDEES 2021 country zip.

Reads the "Stock of vehicles - total (vehicles)" rows of sheet `TrRoad_act`
(codes `Stock.number.{CC}.Tr.Road.*`).
"""

import sys

import pandas as pd
from _jrc_idees import read_road_activity, select_categories
from _schemas import NationalStockSchema

PREFIX = "Stock.number"


def extract_stock(
    activity: pd.DataFrame, jrc_code: str, country_id: str, source: str
) -> pd.DataFrame:
    """Select the vehicle stock rows per category and return them in long format.

    Raises if a category code is missing or if the categories do not add up to
    the total road vehicle stock.
    """
    long = select_categories(activity, PREFIX, jrc_code, country_id)
    long["stock"] = long.pop("value")
    long["country_id"] = country_id
    long["unit"] = "vehicles"
    long["source"] = source
    long = long[list(NationalStockSchema.columns)]
    long = long.sort_values(["year", "vehicle_category"], ignore_index=True)
    return NationalStockSchema.validate(long)


def main(zip_path, jrc_code, country_id, source, output_path):
    """Extract one country and write it as parquet."""
    activity = read_road_activity(zip_path, jrc_code)
    stock = extract_stock(activity, jrc_code, country_id, source)
    stock.to_parquet(output_path, index=False)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
    main(
        zip_path=snakemake.input.zip,  # noqa: F821
        jrc_code=snakemake.params.jrc_code,  # noqa: F821
        country_id=snakemake.wildcards.country,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        output_path=snakemake.output.stock,  # noqa: F821
    )
