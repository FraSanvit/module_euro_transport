"""Fill gaps in a combined national table (vehicle-km or vehicle stock).

Step 5 of the workflow, applied in this order:
1. Gaps inside a series (missing years between two observed years of the same country,
   category and source) are filled by linear interpolation, if at most
   `max_years` consecutive years are missing. Longer gaps are left missing and logged.
2. Stock only: a category with no data at all for a country is filled as the
   country's own passenger-car stock x the mean ratio (category stock / car stock)
   of its neighbour countries in the same year. Outside a neighbour's years its
   first or last ratio is held. Years without own car stock stay missing.

Neighbours are processed even when not configured; only configured countries are
written.

Observed values are never changed. Every row gets a `filled` label ("observed" or the
fill method) and every filled value is logged.
"""

import logging

import numpy as np
import pandas as pd
from _schemas import VEHICLE_CATEGORIES, FilledStockSchema, FilledVkmSchema

LOGGER = logging.getLogger(__name__)
SCHEMAS = {"vkm": FilledVkmSchema, "stock": FilledStockSchema}
OBSERVED = "observed"
INTERPOLATED = "interpolated"
NEIGHBOUR_RATIO = "neighbour_ratio"
ANCHOR = "passenger_car"


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


def neighbour_ratios(
    table: pd.DataFrame, neighbours: list[str], category: str, years: pd.Index
) -> pd.Series:
    """Mean ratio of `category` to car stock over the neighbours, per year in `years`.

    Each neighbour's ratio series is held constant before its first and after its
    last year. Neighbours without the category are skipped.
    """
    wide = table.pivot_table(
        index="year", columns=["country_id", "vehicle_category"], values="stock"
    )
    ratios = {}
    for neighbour in neighbours:
        if (neighbour, category) not in wide or (neighbour, ANCHOR) not in wide:
            LOGGER.warning("Neighbour %s has no %s or %s stock (skipped).", neighbour, category, ANCHOR)
            continue
        ratio = (wide[(neighbour, category)] / wide[(neighbour, ANCHOR)]).dropna()
        if ratio.empty:
            continue
        full = ratio.reindex(ratio.index.union(years)).sort_index().ffill().bfill()
        ratios[neighbour] = full.reindex(years)
    if not ratios:
        return pd.Series(dtype=float)
    return pd.DataFrame(ratios).mean(axis=1)


def fill_missing_categories(
    table: pd.DataFrame, countries: list[str], neighbours: dict
) -> pd.DataFrame:
    """Add stock rows for categories a country lacks entirely (see module docstring)."""
    added = []
    for country in countries:
        if country not in neighbours:
            continue
        own = table[table["country_id"] == country]
        cars = own[own["vehicle_category"] == ANCHOR].set_index("year")
        missing = [c for c in VEHICLE_CATEGORIES if c not in set(own["vehicle_category"])]
        if missing and cars.empty:
            LOGGER.warning("%s: no %s stock, %s left missing.", country, ANCHOR, missing)
            continue
        names = sorted(neighbours[country])
        for category in missing:
            ratio = neighbour_ratios(table, names, category, cars.index)
            if ratio.empty:
                LOGGER.warning("%s %s: no neighbour ratio (left missing).", country, category)
                continue
            stock = cars["stock"] * ratio
            for year, value in stock.items():
                LOGGER.info(
                    "%s %s %d: stock = %.6g (cars x neighbour ratio %.4g).",
                    country, category, year, value, ratio[year],
                )  # fmt: skip
                added.append(
                    {
                        "country_id": country,
                        "year": int(year),
                        "vehicle_category": category,
                        "stock": float(value),
                        "unit": "vehicles",
                        "source": f"{ANCHOR} stock x neighbour ratio ({', '.join(names)})",
                        "filled": NEIGHBOUR_RATIO,
                    }
                )
    if not added:
        return table
    added = pd.DataFrame(added)[table.columns].astype(table.dtypes.to_dict())
    return pd.concat([table, added], ignore_index=True)


def fill(
    table: pd.DataFrame,
    metric: str,
    max_years: int,
    countries: list[str] | None = None,
    neighbours: dict | None = None,
) -> pd.DataFrame:
    """Apply all gap-filling sub-steps, keep `countries` (default: all), sort and validate."""
    countries = sorted(table["country_id"].unique()) if countries is None else countries
    result = interpolate_gaps(table, metric, max_years)
    if metric == "stock":
        result = fill_missing_categories(result, countries, neighbours or {})
    result = result[result["country_id"].isin(countries)]
    counts = result["filled"].value_counts().to_dict()
    LOGGER.info("%s rows by fill method: %s", metric, counts)
    result = result.sort_values(
        ["country_id", "year", "vehicle_category"], ignore_index=True
    )
    return SCHEMAS[metric].validate(result)


def main(input_path, output_path, metric, max_years, countries, neighbours):
    """Read the combined table, fill gaps and write parquet for `countries`."""
    filled = fill(pd.read_parquet(input_path), metric, max_years, countries, neighbours)
    missing = sorted(set(countries) - set(filled["country_id"]))
    if missing:
        LOGGER.warning("No %s for configured countries %s (left missing).", metric, missing)
    filled.to_parquet(output_path, index=False)


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
        countries=snakemake.params.countries,  # noqa: F821
        neighbours=snakemake.params.neighbours,  # noqa: F821
    )
