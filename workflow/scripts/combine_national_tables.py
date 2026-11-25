"""Combine national tables (vehicle-km or vehicle stock) into one validated table.

Vehicle-km: heavy_duty comes from the heavy-duty level chain (filled national levels,
territorial, MULTIYEAR_PLAN.md §0.2) when its table is given; the per-source heavy_duty
rows are then dropped and logged against the new values.
"""

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


HEAVY_DUTY = "heavy_duty"
HDV_SOURCE = "HDV level chain"


def heavy_duty_rows(filled: pd.DataFrame) -> pd.DataFrame:
    """National vkm rows (territorial totals) from the filled heavy-duty levels.

    The source names the fill methods where any of the three filled series was not
    observed, e.g. "HDV level chain (filled: neighbours_gdp_pps, neighbour_mean)".
    """
    labels = filled[["filled_L_dom", "filled_L_int", "filled_alpha_E"]]
    methods = labels.apply(lambda row: sorted({v for v in row if v != "observed"}), axis=1)
    source = [HDV_SOURCE + (f" (filled: {', '.join(m)})" if m else "") for m in methods]
    return pd.DataFrame(
        {
            "country_id": filled["country_id"].to_numpy(),
            "year": filled["year"].astype(int).to_numpy(),
            "vehicle_category": HEAVY_DUTY,
            "vkm": filled["total"].astype(float).to_numpy(),
            "unit": "vkm",
            "source": source,
            "principle": "territorial",
        }
    )


def replace_heavy_duty(frames: list[pd.DataFrame], filled: pd.DataFrame, countries: list[str]) -> list[pd.DataFrame]:
    """Drop per-source heavy_duty rows and add the level-chain rows for `countries`."""
    hdv = heavy_duty_rows(filled[filled["country_id"].isin(countries)])
    old = pd.concat(frames, ignore_index=True)
    old = old[old["vehicle_category"] == HEAVY_DUTY].set_index(["country_id", "year"])["vkm"]
    new = hdv.set_index(["country_id", "year"])["vkm"]
    for country in sorted(set(old.index.get_level_values(0)) | set(new.index.get_level_values(0))):
        before, after = old.get((country, 2019), float("nan")), new.get((country, 2019), float("nan"))
        LOGGER.info("%s heavy_duty 2019: %.4g -> %.4g vkm (HDV level chain).", country, before, after)
    kept = [frame[frame["vehicle_category"] != HEAVY_DUTY] for frame in frames]
    return [*kept, hdv]


def log_missing_countries(combined: pd.DataFrame, countries: list[str]) -> list[str]:
    """Log configured countries that have no data in the combined table."""
    missing = sorted(set(countries) - set(combined["country_id"]))
    if missing:
        LOGGER.warning("No data for configured countries %s (left missing).", missing)
    return missing


def main(input_paths, output_path, metric, countries, heavy_duty_path=None, processed=None):
    """Read, combine and write parquet tables."""
    frames = [pd.read_parquet(p) for p in input_paths]
    if heavy_duty_path:
        frames = replace_heavy_duty(frames, pd.read_parquet(heavy_duty_path), processed or countries)
    combined = combine(frames, SCHEMAS[metric])
    duplicated = combined.duplicated(["country_id", "year", "vehicle_category"])
    if duplicated.any():
        raise ValueError("A country has values from more than one source.")
    log_missing_countries(combined, countries)
    combined.to_parquet(output_path, index=False)


def as_list(paths) -> list[str]:
    """A single path (named input with one file) or a list of paths, as a list."""
    return [paths] if isinstance(paths, str) else list(paths)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        input_paths=as_list(snakemake.input.get("tables", list(snakemake.input))),  # noqa: F821
        output_path=snakemake.output[0],  # noqa: F821
        metric=snakemake.params.metric,  # noqa: F821
        countries=snakemake.params.countries,  # noqa: F821
        heavy_duty_path=snakemake.input.get("heavy_duty"),  # noqa: F821
        processed=snakemake.params.get("processed"),  # noqa: F821
    )
