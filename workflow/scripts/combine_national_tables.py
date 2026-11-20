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


def select_by_priority(combined: pd.DataFrame, priority: list[str]) -> pd.DataFrame:
    """Keep, per country, year and category, the value of the first source in `priority`.

    Logs which sources were used per country and the year-to-year change wherever
    the selected source switches, since sources may not be consistent.
    """
    unknown = sorted(set(combined["source"]) - set(priority))
    if unknown:
        raise ValueError(f"Sources without a priority: {unknown}.")
    rank = combined["source"].map({source: i for i, source in enumerate(priority)})
    selected = (
        combined.assign(_rank=rank)
        .sort_values("_rank", kind="stable")
        .drop_duplicates(["country_id", "year", "vehicle_category"])
        .drop(columns="_rank")
        .sort_values(["country_id", "year", "vehicle_category"], ignore_index=True)
    )
    for country, group in selected.groupby("country_id"):
        counts = group["source"].value_counts().to_dict()
        LOGGER.info("%s: values per source %s", country, counts)
    for switch in source_switches(selected).itertuples():
        LOGGER.warning(
            "%s %s: source switches %s -> %s at %d (%.0f -> %.0f, ratio %.2f).",
            switch.country_id,
            switch.vehicle_category,
            switch.prev_source,
            switch.source,
            switch.year,
            switch.prev_value,
            switch.value,
            switch.ratio,
        )
    return selected


def source_switches(table: pd.DataFrame) -> pd.DataFrame:
    """Consecutive years of a country-category where the source changes."""
    value = "vkm" if "vkm" in table else "stock"
    table = table.sort_values(["country_id", "vehicle_category", "year"])
    group = table.groupby(["country_id", "vehicle_category"])
    prev = group[["year", "source", value]].shift()
    switch = table["source"].ne(prev["source"]) & table["year"].eq(prev["year"] + 1)
    out = table.loc[switch, ["country_id", "vehicle_category", "year", "source"]]
    out["prev_source"] = prev.loc[switch, "source"]
    out["prev_value"] = prev.loc[switch, value]
    out["value"] = table.loc[switch, value]
    out["ratio"] = out["value"] / out["prev_value"]
    return out.reset_index(drop=True)


def main(input_paths, output_path, metric, countries, priority=None):
    """Read, combine (optionally select by source priority) and write parquet tables."""
    combined = combine([pd.read_parquet(p) for p in input_paths], SCHEMAS[metric])
    if priority:
        combined = select_by_priority(combined, priority)
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
        priority=snakemake.params.get("priority"),  # noqa: F821
    )
