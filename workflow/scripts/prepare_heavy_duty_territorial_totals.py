"""Territorial national totals of heavy-duty vkm (MULTIYEAR_PLAN.md §6.8).

- GBR: DfT road traffic by region, all HGVs, vehicle-miles summed over GB regions,
  converted to km and scaled to the United Kingdom with the GB share of UK
  population (Eurostat; latest year with both UK and Northern Ireland, held).
- SWE: Trafikanalys t04010, heavy lorries (3 weight classes), million vkm.
- CHE: BFS heavy goods vehicles on Swiss territory (module rule prepare_bfs_road_vkm).
All include foreign vehicles. Returns `country_id, year, T, source`.
"""

import json
import logging

import pandas as pd
from _heavy_duty import MIO, read_eurostat

LOGGER = logging.getLogger(__name__)


def dft_totals(path, miles_to_km: float) -> pd.Series:
    """GB HGV vkm per year (sum of `all_HGVs` over regions)."""
    table = pd.read_csv(path)
    return table.groupby("year")["all_HGVs"].sum(min_count=1) * miles_to_km


def uk_scaling(path, covered_part: str) -> float:
    """UK / GB population, latest year with UK and the uncovered part (UKN)."""
    pop = read_eurostat(path, ["geo"]).pivot(index="year", columns="geo", values="value")
    pop = pop[["UK", covered_part]].dropna()
    year = int(pop.index.max())
    factor = pop.loc[year, "UK"] / (pop.loc[year, "UK"] - pop.loc[year, covered_part])
    LOGGER.info("GBR: GB -> UK factor %.4f from population %d (held for all years).", factor, year)
    return float(factor)


def trafa_totals(path) -> pd.Series:
    """Swedish heavy-lorry vkm per year (sum of the weight classes)."""
    rows = json.loads(open(path, encoding="utf-8").read())["Rows"]
    records = []
    for row in rows:
        cells = {cell["Column"]: cell for cell in row["Cell"]}
        value = cells["fordonkm"]["Value"]
        if value in ("", None):
            continue
        records.append({"year": int(cells["ar"]["Name"]), "vkm": float(value) * MIO})
    table = pd.DataFrame(records)
    counts = table.groupby("year").size()
    complete = counts[counts == counts.max()].index
    return table[table["year"].isin(complete)].groupby("year")["vkm"].sum()


def bfs_totals(path) -> pd.Series:
    """Swiss heavy goods vehicle vkm per year from the module's BFS table."""
    table = pd.read_parquet(path)
    heavy = table[(table["country_id"] == "CHE") & (table["vehicle_category"] == "heavy_duty")]
    return heavy.set_index("year")["vkm"]


def main(dft_path, uk_population_path, trafa_path, bfs_path, settings, countries, years, output_path):
    """Collect the territorial totals of the processed override countries."""
    frames = []
    if "GBR" in countries:
        gb = dft_totals(dft_path, settings["dft"]["miles_to_km"])
        uk = gb * uk_scaling(uk_population_path, settings["dft"]["covered_part"])
        frames.append(pd.DataFrame({"country_id": "GBR", "year": uk.index, "T": uk.to_numpy(), "source": settings["dft"]["source"]}))
    if "SWE" in countries:
        se = trafa_totals(trafa_path)
        frames.append(pd.DataFrame({"country_id": "SWE", "year": se.index, "T": se.to_numpy(), "source": settings["trafa"]["source"]}))
    if "CHE" in countries and bfs_path:
        ch = bfs_totals(bfs_path)
        frames.append(pd.DataFrame({"country_id": "CHE", "year": ch.index, "T": ch.to_numpy(), "source": "BFS je-d-11.05.01.01"}))
    columns = ["country_id", "year", "T", "source"]
    totals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    totals = totals[totals["year"].isin(years)].astype({"year": int, "T": float})
    for country, group in totals.groupby("country_id"):
        LOGGER.info("%s territorial total: years %d-%d", country, group["year"].min(), group["year"].max())
    totals.sort_values(["country_id", "year"], ignore_index=True).to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        dft_path=snakemake.input.dft,  # noqa: F821
        uk_population_path=snakemake.input.uk_population,  # noqa: F821
        trafa_path=snakemake.input.trafa,  # noqa: F821
        bfs_path=snakemake.input.get("bfs"),  # noqa: F821
        settings=snakemake.params.settings,  # noqa: F821
        countries=snakemake.params.countries,  # noqa: F821
        years=snakemake.params.years,  # noqa: F821
        output_path=snakemake.output.totals,  # noqa: F821
    )
