"""Combine per-country national vehicle-km tables into one validated table."""

import sys

import pandas as pd
from _schemas import NationalVkmSchema


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, sort and validate national vehicle-km tables."""
    if not frames:
        raise ValueError("Nothing to combine.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["country_id", "year", "vehicle_category", "source"], ignore_index=True
    )
    return NationalVkmSchema.validate(combined)


def main(input_paths, output_path):
    """Read, combine and write parquet tables."""
    combined = combine([pd.read_parquet(p) for p in input_paths])
    combined.to_parquet(output_path, index=False)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
    main(snakemake.input.vkm, snakemake.output.vkm)  # noqa: F821
