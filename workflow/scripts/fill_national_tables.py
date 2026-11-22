"""Fill gaps in a combined national table (vehicle-km or vehicle stock).

Step 5 of the workflow; currently only the first sub-step:
- Gaps inside a series (missing years between two observed years of the same country,
  category and source) are filled by linear interpolation, if at most
  `max_years` consecutive years are missing. Longer gaps are left missing and logged.

Observed values are never changed. Every row gets a `filled` label ("observed" or the
fill method) and every filled value is logged.
"""

import logging

import numpy as np
import pandas as pd
from _schemas import FilledStockSchema, FilledVkmSchema

LOGGER = logging.getLogger(__name__)
SCHEMAS = {"vkm": FilledVkmSchema, "stock": FilledStockSchema}
OBSERVED = "observed"
INTERPOLATED = "interpolated"


def interpolate_gaps(table: pd.DataFrame, metric: str, max_years: int) -> pd.DataFrame:
    """Return the table with a `filled` column and interpolated inner gaps added.

    A gap is interpolated only if both neighbouring observed years share every
    column except year and value (same unit, source and, for vehicle-km, principle).
    """
    labels = [col for col in table.columns if col not in ("year", metric)]
    filled = []
    for _, series in table.groupby(["country_id", "vehicle_category"], sort=True):
        series = series.sort_values("year")
        rows = series.to_dict("records")
        for before, after in zip(rows[:-1], rows[1:]):
            missing = np.arange(before["year"] + 1, after["year"])
            if missing.size == 0:
                continue
            name = f"{before['country_id']} {before['vehicle_category']}"
            span = f"{missing[0]}-{missing[-1]}" if missing.size > 1 else f"{missing[0]}"
            if missing.size > max_years:
                LOGGER.warning(
                    "%s: years %s not interpolated, gap of %d years > %d (left missing).",
                    name, span, missing.size, max_years,
                )  # fmt: skip
                continue
            if any(before[col] != after[col] for col in labels):
                LOGGER.warning(
                    "%s: years %s not interpolated, %s and %s differ in %s (left missing).",
                    name, span, before["year"], after["year"], labels,
                )  # fmt: skip
                continue
            values = np.interp(
                missing, [before["year"], after["year"]], [before[metric], after[metric]]
            )
            for year, value in zip(missing, values):
                LOGGER.info("%s %d: %s = %.6g interpolated.", name, year, metric, value)
                filled.append(
                    {**before, "year": int(year), metric: float(value), "filled": INTERPOLATED}
                )
    observed = table.assign(filled=OBSERVED)
    if not filled:
        return observed.reset_index(drop=True)
    added = pd.DataFrame(filled)[observed.columns].astype(observed.dtypes.to_dict())
    return pd.concat([observed, added], ignore_index=True)


def fill(table: pd.DataFrame, metric: str, max_years: int) -> pd.DataFrame:
    """Apply all gap-filling sub-steps, sort and validate."""
    result = interpolate_gaps(table, metric, max_years)
    counts = result["filled"].value_counts().to_dict()
    LOGGER.info("%s rows by fill method: %s", metric, counts)
    result = result.sort_values(
        ["country_id", "year", "vehicle_category"], ignore_index=True
    )
    return SCHEMAS[metric].validate(result)


def main(input_path, output_path, metric, max_years):
    """Read the combined table, fill gaps and write parquet."""
    fill(pd.read_parquet(input_path), metric, max_years).to_parquet(
        output_path, index=False
    )


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        input_path=snakemake.input[0],  # noqa: F821
        output_path=snakemake.output[0],  # noqa: F821
        metric=snakemake.params.metric,  # noqa: F821
        max_years=snakemake.params.max_years,  # noqa: F821
    )
