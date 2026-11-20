"""Combine national tables (vehicle-km or vehicle stock) into one validated table."""

import logging

import pandas as pd
import pandera.pandas as pa
from _schemas import NationalStockSchema, NationalVkmSchema

LOGGER = logging.getLogger(__name__)
SCHEMAS = {"vkm": NationalVkmSchema, "stock": NationalStockSchema}


def combine(
    frames: list[pd.DataFrame], schema: pa.DataFrameSchema = NationalVkmSchema
) -> pd.DataFrame:
    """Concatenate, sort and validate national tables."""
    if not frames:
        raise ValueError("Nothing to combine.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["country_id", "year", "vehicle_category", "source"], ignore_index=True
    )
    return schema.validate(combined)


def log_missing_countries(combined: pd.DataFrame, countries: list[str]) -> list[str]:
    """Log configured countries that have no data in the combined table."""
    missing = sorted(set(countries) - set(combined["country_id"]))
    if missing:
        LOGGER.warning("No data for configured countries %s (left missing).", missing)
    return missing


def main(input_paths, output_path, metric, countries):
    """Read, combine and write parquet tables."""
    combined = combine([pd.read_parquet(p) for p in input_paths], SCHEMAS[metric])
    log_missing_countries(combined, countries)
    combined.to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        input_paths=list(snakemake.input),  # noqa: F821
        output_path=snakemake.output[0],  # noqa: F821
        metric=snakemake.params.metric,  # noqa: F821
        countries=snakemake.params.countries,  # noqa: F821
    )
