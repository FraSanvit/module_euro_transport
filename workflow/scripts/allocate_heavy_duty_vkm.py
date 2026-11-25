"""Heavy-duty vkm rasters for one year: national levels spread over their keys (MULTIYEAR_PLAN.md §6.10).

vkm[cell, component] = level_c,component x key_cell / sum of the key over the cells of c.
Keys: international -> ETISplus international (E-roads + first and last legs);
domestic long-haul -> ETISplus domestic (for countries ETISplus does not cover: the
international key); domestic short-haul -> NUTS-3 tonnes x population. A key that sums
to zero in a country falls back to the short-haul key, then to an even spread (e.g.
islands without a network: CYP, ISL, MLT). Cells
belong to the country containing their centre. Conservation is checked per country.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from _raster import read_raster, write_raster
from _short_haul import short_haul_key
from prepare_heavy_duty_etisplus_keys import population_for_year

LOGGER = logging.getLogger(__name__)
COMPONENTS = {"international": "L_int", "domestic_long": "L_long", "domestic_short": "L_short"}
NOT_COVERED = "etisplus_not_covering"
TOLERANCE = 1e-6


def allocate(level: np.ndarray, keys: list[np.ndarray], orders: list[list[int]], ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spread `level` (per country) over the first key of the country's order with a positive sum.

    `orders[c]` lists indices into `keys` for country c. Returns the raster and the key
    index used per country.
    """
    valid = ids >= 0
    n = len(level)
    sums = np.array([np.bincount(ids[valid], weights=k[valid], minlength=n) for k in keys])
    used = np.array([next((i for i in order if sums[i, c] > 0), order[-1]) for c, order in enumerate(orders)])
    out = np.zeros(ids.shape)
    key_values = np.choose(used[ids[valid]], [k[valid] for k in keys])
    denominator = sums[used, np.arange(n)][ids[valid]]
    out[valid] = np.where(denominator > 0, level[ids[valid]] * key_values / np.where(denominator > 0, denominator, 1), 0.0)
    return out, used


def main(
    levels_path, grid_path, polygons_path, population_paths: dict, nuts_rasters: dict, nuts_codes_path,
    tonnes_path, key_paths: dict, codes, year: int, output_paths: dict, report_path,
):  # fmt: skip
    """Write the three component rasters of `year` and the conservation report."""
    ids, profile = read_raster(grid_path)
    ids = ids.astype("int32")
    countries = gpd.read_parquet(polygons_path)["country_id"].tolist()
    levels = pd.read_parquet(levels_path)
    levels = levels[levels["year"] == year].set_index("country_id").reindex(countries)
    missing = levels.index[levels["total"].isna()].tolist()
    if missing:
        raise ValueError(f"No heavy-duty level for {missing} in {year}.")
    region_rasters = {str(v): read_raster(p)[0] for v, p in nuts_rasters.items()}
    nuts_codes = pd.read_parquet(nuts_codes_path)
    region_codes = {v: g.sort_values("index")["nuts"].tolist() for v, g in nuts_codes.groupby("version")}
    population = population_for_year(population_paths, year)
    short, report = short_haul_key(
        population, ids, countries, {c: codes[c] for c in countries}, pd.read_parquet(tonnes_path),
        region_rasters, region_codes, year,
    )  # fmt: skip
    for row in report.itertuples():
        LOGGER.info("%s %d short-haul key: %s (data year %s, NUTS %s).", row.country_id, year, row.method, row.data_year, row.version)
    k = {name: read_raster(path)[0].astype("float64") for name, path in key_paths.items()}
    key_int = k["network_int"] + k["legs_int"]
    key_dom = k["network_dom"] + k["legs_dom"]
    even = (ids >= 0).astype("float64")
    not_covered = levels["flags"].fillna("").str.contains(NOT_COVERED).to_numpy()
    n = len(countries)
    plans = {
        "international": ([key_int, short, even], [[0, 1, 2]] * n),
        "domestic_long": ([key_dom, key_int, short, even], [[1, 2, 3] if nc else [0, 2, 3] for nc in not_covered]),
        "domestic_short": ([short, even], [[0, 1]] * n),
    }
    names = {
        "international": ["etisplus_international", "short_haul", "even"],
        "domestic_long": ["etisplus_domestic", "etisplus_international", "short_haul", "even"],
        "domestic_short": ["short_haul", "even"],
    }
    rows = []
    valid = ids >= 0
    for component, (keys, orders) in plans.items():
        level = levels[COMPONENTS[component]].to_numpy(dtype="float64")
        raster, used = allocate(level, keys, orders, ids)
        raster32 = raster.astype("float32")
        sums = np.bincount(ids[valid], weights=raster32[valid].astype("float64"), minlength=len(countries))
        for c, country in enumerate(countries):
            error = abs(sums[c] - level[c]) / level[c] if level[c] > 0 else abs(sums[c])
            key_name = names[component][used[c]]
            if used[c] != orders[c][0] and level[c] > 0:
                LOGGER.warning("%s %d %s: key %s sums to zero, %s used.", country, year, component,
                               names[component][orders[c][0]], key_name)  # fmt: skip
            rows.append({"country_id": country, "year": year, "component": component, "level": level[c],
                         "raster_sum": sums[c], "relative_error": error, "key": key_name})  # fmt: skip
        write_raster(output_paths[component], raster32, profile["transform"], profile["crs"])
    report_table = pd.DataFrame(rows)
    worst = report_table["relative_error"].max()
    LOGGER.info("%d: largest relative conservation error %.2e.", year, worst)
    if worst > TOLERANCE:
        raise ValueError(f"Conservation error {worst:.2e} > {TOLERANCE} in {year}.")
    report_table.to_parquet(report_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(filename=snakemake.log[0], level=logging.INFO, format="%(levelname)s %(message)s")  # noqa: F821
    epochs = snakemake.params.epochs  # noqa: F821
    versions = snakemake.params.versions  # noqa: F821
    main(
        levels_path=snakemake.input.levels,  # noqa: F821
        grid_path=snakemake.input.grid,  # noqa: F821
        polygons_path=snakemake.input.polygons,  # noqa: F821
        population_paths={e: snakemake.input[f"population_{e}"] for e in epochs},  # noqa: F821
        nuts_rasters={v: snakemake.input[f"nuts_{v}"] for v in versions},  # noqa: F821
        nuts_codes_path=snakemake.input.nuts_codes,  # noqa: F821
        tonnes_path=snakemake.input.tonnes,  # noqa: F821
        key_paths={k: snakemake.input[k] for k in ["network_dom", "network_int", "legs_dom", "legs_int"]},  # noqa: F821
        codes=snakemake.params.codes,  # noqa: F821
        year=int(snakemake.wildcards.year),  # noqa: F821
        output_paths={c: snakemake.output[c] for c in COMPONENTS},  # noqa: F821
        report_path=snakemake.output.report,  # noqa: F821
    )
