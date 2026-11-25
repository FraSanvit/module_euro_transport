"""Short-haul key: freight-weighted population (MULTIYEAR_PLAN.md §6.10).

P_cell,t = pop_cell,t x q_n,t with q_n,t = tonnes_n,t / pop_n,t, tonnes = (loaded +
unloaded) / 2 of NUTS-3 region n (road_go_na_rl3g / ru3g). Within a region the key
follows population; between regions it follows tonnes. Per country, the tonnes of year
t are used, else the nearest earlier year, else the nearest later year; they are matched
to the NUTS version holding most of their codes. Cells without a usable region get the
country's mean q (population only if the country has no NUTS-3 tonnes).
"""

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def tonnes_table(loaded: pd.DataFrame, unloaded: pd.DataFrame) -> pd.DataFrame:
    """`nuts, year, tonnes` = mean of loaded and unloaded (one side if only one is reported).

    Inputs have `geo, year, value` (thousand tonnes, all goods). Only NUTS-3 codes
    (5 characters, extra-regio `..ZZZ` excluded) are kept.
    """
    def nuts3(t):
        t = t[(t["geo"].str.len() == 5) & ~t["geo"].str.endswith("ZZZ")]
        return t.set_index(["geo", "year"])["value"]

    both = pd.DataFrame({"loaded": nuts3(loaded), "unloaded": nuts3(unloaded)})
    out = both.mean(axis=1, skipna=True).rename("tonnes").reset_index()
    out = out.rename(columns={"geo": "nuts"})
    out["tonnes"] = out["tonnes"] * 1000
    return out.dropna().sort_values(["nuts", "year"], ignore_index=True)


def data_year(years: list[int], year: int):
    """The year of `years` to use for `year`: itself, else nearest earlier, else nearest later."""
    if not years:
        return None
    earlier = [y for y in years if y <= year]
    return max(earlier) if earlier else min(years)


def choose_version(codes: set, versions: dict) -> tuple:
    """NUTS version (key of `versions`: code -> index) holding most of `codes`; (version, unmatched)."""
    best = max(versions, key=lambda v: (len(codes & set(versions[v])), v))
    return best, sorted(codes - set(versions[best]))


def short_haul_key(
    population: np.ndarray,
    country_ids: np.ndarray,
    countries: list[str],
    nuts_prefix: dict,
    tonnes: pd.DataFrame,
    region_rasters: dict,
    region_codes: dict,
    year: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Short-haul key raster for `year` and a per-country report.

    `country_ids`: country index per cell (-1 outside). `nuts_prefix`: ISO3 -> NUTS country
    prefix (EL, UK, ...). `region_rasters`: version -> region index raster;
    `region_codes`: version -> list of NUTS codes by index.
    """
    key = population.astype("float64").copy()
    lookup = {v: dict(zip(codes, range(len(codes)))) for v, codes in region_codes.items()}
    rows = []
    for idx, country in enumerate(countries):
        cells = country_ids == idx
        prefix = nuts_prefix.get(country)
        mine = tonnes[tonnes["nuts"].str.startswith(prefix)] if prefix else tonnes.iloc[0:0]
        used = data_year(sorted(mine["year"].unique()), year)
        if used is None:
            rows.append({"country_id": country, "data_year": np.nan, "version": "", "unmatched": "", "method": "population"})
            continue
        values = mine[mine["year"] == used].set_index("nuts")["tonnes"]
        version, unmatched = choose_version(set(values.index), lookup)
        regions = region_rasters[version]
        ids = np.where(cells, regions, -1)
        n = len(region_codes[version])
        valid = ids >= 0
        pop_region = np.bincount(ids[valid], weights=population[valid], minlength=n)
        tonnes_region = np.full(n, np.nan)
        for code, value in values.items():
            if code in lookup[version]:
                tonnes_region[lookup[version][code]] = value
        q_region = np.where(pop_region > 0, tonnes_region / np.where(pop_region > 0, pop_region, 1), np.nan)
        q_cell = np.full(population.shape, np.nan)
        q_cell[valid] = q_region[ids[valid]]
        has_q = cells & np.isfinite(q_cell)
        weight = population[has_q].sum()
        q_mean = (population[has_q] * q_cell[has_q]).sum() / weight if weight > 0 else 1.0
        q_cell[cells & ~np.isfinite(q_cell)] = q_mean
        key[cells] = population[cells] * q_cell[cells]
        share_fallback = population[cells & ~has_q].sum() / max(population[cells].sum(), 1e-12)
        if unmatched:
            LOGGER.warning("%s %d: NUTS-3 codes %s not in NUTS %s (country mean q).", country, used, unmatched, version)
        rows.append({
            "country_id": country, "data_year": used, "version": version, "unmatched": ";".join(unmatched),
            "method": "nuts3_tonnes", "population_share_country_mean_q": share_fallback,
        })  # fmt: skip
    return key, pd.DataFrame(rows)
