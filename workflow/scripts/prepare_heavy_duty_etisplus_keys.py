"""ETISplus allocation keys on the grid (MULTIYEAR_PLAN.md §6.1 steps 2 and 4, §6.10).

E-road vkm: each edge is a straight line between its nodes, cut into pieces of at most a
quarter cell; each piece adds flow x Distance / n_pieces to the cell of its midpoint,
separately for domestic and international OD flows. Manually added links with more than
`ferry_sea_share` of their pieces off land are ferries and dropped (no road km).

First and last legs: each ETISplus zone is matched to the NUTS-3 region containing its
centre (NUTS versions tried in turn) and its legs are spread over that region by the
short-haul key of the reference year. Zones without a region are spread over their
country. Writes four rasters: E-road and legs, domestic and international [vkm].
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from _raster import read_raster, write_raster
from _short_haul import short_haul_key
from pyproj import Transformer

LOGGER = logging.getLogger(__name__)


def population_for_year(epochs: dict, year: int) -> np.ndarray:
    """Linear interpolation between GHS-POP epochs (paths by year); nearest epoch outside."""
    years = sorted(epochs)
    if year <= years[0] or year >= years[-1]:
        return read_raster(epochs[years[0] if year <= years[0] else years[-1]])[0].astype("float64")
    lower = max(y for y in years if y <= year)
    upper = min(y for y in years if y >= year)
    if lower == upper:
        return read_raster(epochs[lower])[0].astype("float64")
    w = (year - lower) / (upper - lower)
    return (1 - w) * read_raster(epochs[lower])[0].astype("float64") + w * read_raster(epochs[upper])[0].astype("float64")


def edge_pieces(x0, y0, x1, y1, max_piece: float):
    """Midpoints of equal pieces (at most `max_piece` long) of each segment; edge index per piece."""
    length = np.hypot(x1 - x0, y1 - y0)
    n = np.maximum(np.ceil(length / max_piece).astype(int), 1)
    edge = np.repeat(np.arange(len(n)), n)
    first = np.repeat(np.cumsum(n) - n, n)
    t = (np.arange(n.sum()) - first + 0.5) / n[edge]
    return x0[edge] + t * (x1 - x0)[edge], y0[edge] + t * (y1 - y0)[edge], edge, n


def cell_index(x, y, transform, shape):
    """Row and column of points; -1 where outside the grid."""
    col = np.floor((x - transform.c) / transform.a).astype(int)
    row = np.floor((y - transform.f) / transform.e).astype(int)
    inside = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    return np.where(inside, row, -1), np.where(inside, col, -1)


def rasterise_network(edges: pd.DataFrame, nodes: pd.DataFrame, land: np.ndarray, transform, crs, ferry_sea_share: float):
    """E-road vkm rasters (domestic, international) and the number of ferry links dropped."""
    shape = land.shape
    xy = nodes.set_index("Network_Node_ID")[["Network_Node_X", "Network_Node_Y"]]
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ax, ay = to_grid.transform(*xy.reindex(edges["Network_Node_A_ID"]).to_numpy().T)
    bx, by = to_grid.transform(*xy.reindex(edges["Network_Node_B_ID"]).to_numpy().T)
    px, py, edge, n = edge_pieces(np.asarray(ax), np.asarray(ay), np.asarray(bx), np.asarray(by), abs(transform.a) / 4)
    row, col = cell_index(px, py, transform, shape)
    on_land = (row >= 0) & land[np.maximum(row, 0), np.maximum(col, 0)]
    sea_share = 1 - np.bincount(edge, weights=on_land, minlength=len(n)) / n
    ferry = (edges["Manually_Added"].to_numpy() == 1) & (sea_share > ferry_sea_share)
    keep = (row >= 0) & ~ferry[edge]
    out = {}
    for kind in ["dom", "int"]:
        vkm = (edges[f"flow_{kind}"].to_numpy() * edges["Distance"].to_numpy())[edge] / n[edge]
        raster = np.zeros(shape)
        np.add.at(raster, (row[keep], col[keep]), vkm[keep])
        out[kind] = raster
        LOGGER.info(
            "E-road %s: %.4g vkm on the grid, %.4g on ferry links dropped, %.4g outside the grid.",
            kind, raster.sum(), vkm[ferry[edge]].sum(), vkm[(row < 0) & ~ferry[edge]].sum(),
        )  # fmt: skip
    return out, int(ferry.sum())


def zone_regions(legs: pd.DataFrame, transform, shape, crs, region_rasters: dict, versions: list, country_ids=None, countries=None):
    """Region (version, index) of each zone's centre, trying `versions` in turn.

    Without a region: ("country", index) if the centre lies in the zone's own country on
    the grid (e.g. BA, no NUTS), else None (outside the grid, e.g. the Canaries: dropped).
    """
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_grid.transform(legs["x"].to_numpy(), legs["y"].to_numpy())
    row, col = cell_index(np.asarray(x), np.asarray(y), transform, shape)
    found = []
    for r, c in zip(row, col):
        hit = None
        if r >= 0:
            for version in versions:
                index = region_rasters[version][r, c]
                if index >= 0:
                    hit = (version, int(index))
                    break
            if hit is None and country_ids is not None and zone_geo_index(legs, len(found), countries) == country_ids[r, c]:
                hit = ("country", int(country_ids[r, c]))
        found.append(hit)
    return found


def zone_geo_index(legs: pd.DataFrame, i: int, countries) -> int:
    """Index of zone i's country in `countries` (-2 if not there)."""
    geo = legs["geo"].iloc[i]
    return countries.index(geo) if countries is not None and geo in countries else -2


def spread_legs(legs, regions, key, country_ids, countries, region_rasters):
    """Legs rasters (dom, int): each zone's legs over its region (else its country) by `key`.

    Regions whose key sums to zero are spread evenly over their cells.
    """
    out = {kind: np.zeros(key.shape) for kind in ["dom", "int"]}
    targets = {}
    dropped = 0.0
    for (_, zone), target in zip(legs.iterrows(), regions):
        if target is None:
            if zone["geo"] in countries:
                dropped += zone["legs_dom"] + zone["legs_int"]
            continue
        amounts = targets.setdefault(target[0], {})
        dom, intl = amounts.get(target[1], (0.0, 0.0))
        amounts[target[1]] = (dom + zone["legs_dom"], intl + zone["legs_int"])
    LOGGER.info("Legs of zones of the scope countries outside the grid (dropped from the key): %.4g vkm.", dropped)
    for version, amounts in targets.items():
        ids = country_ids if version == "country" else region_rasters[version]
        valid = ids >= 0
        n = int(max(ids.max() + 1, max(amounts) + 1))
        weight = np.bincount(ids[valid], weights=key[valid], minlength=n)
        count = np.bincount(ids[valid], minlength=n)
        share = np.where(weight[ids[valid]] > 0, key[valid] / np.where(weight > 0, weight, 1)[ids[valid]],
                         1 / np.maximum(count, 1)[ids[valid]])  # fmt: skip
        lost = sum(d + i for index, (d, i) in amounts.items() if count[index] == 0)
        if lost:
            LOGGER.warning("%s: %.4g vkm of legs in regions without cells (dropped).", version, lost)
        for k, kind in enumerate(["dom", "int"]):
            amount = np.zeros(n)
            for index, values in amounts.items():
                amount[index] = values[k]
            out[kind][valid] += amount[ids[valid]] * share
        if version == "country":
            LOGGER.info("Legs spread over whole countries (zone without NUTS-3 region): %.4g vkm.",
                        sum(d + i for d, i in amounts.values()))  # fmt: skip
    return out


def main(
    edges_path, nodes_path, legs_path, grid_path, polygons_path, population_paths: dict, nuts_rasters: dict,
    nuts_codes_path, tonnes_path, codes, reference_year, ferry_sea_share, output_paths: dict,
):  # fmt: skip
    """Write the E-road and legs rasters."""
    country_ids, profile = read_raster(grid_path)
    transform, crs, shape = profile["transform"], profile["crs"], country_ids.shape
    countries = gpd.read_parquet(polygons_path)["country_id"].tolist()
    network, ferries = rasterise_network(
        pd.read_parquet(edges_path), pd.read_csv(nodes_path), country_ids >= 0, transform, crs, ferry_sea_share
    )
    LOGGER.info("Ferry links dropped: %d.", ferries)
    region_rasters = {str(v): read_raster(p)[0] for v, p in nuts_rasters.items()}
    nuts_codes = pd.read_parquet(nuts_codes_path)
    region_codes = {v: g.sort_values("index")["nuts"].tolist() for v, g in nuts_codes.groupby("version")}
    population = population_for_year(population_paths, reference_year)
    key, _ = short_haul_key(
        population, country_ids, countries, {c: codes[c] for c in countries}, pd.read_parquet(tonnes_path),
        region_rasters, region_codes, reference_year,
    )  # fmt: skip
    legs = pd.read_parquet(legs_path)
    to_iso = {geo: iso for iso, geo in codes.items()}
    legs = legs.assign(geo=legs["geo"].map(to_iso))
    regions = zone_regions(legs, transform, shape, crs, region_rasters, sorted(region_rasters), country_ids, countries)
    spread = spread_legs(legs, regions, key, country_ids, countries, region_rasters)
    for kind in ["dom", "int"]:
        write_raster(output_paths[f"network_{kind}"], network[kind].astype("float32"), transform, crs)
        write_raster(output_paths[f"legs_{kind}"], spread[kind].astype("float32"), transform, crs)
        LOGGER.info("Legs %s on the grid: %.4g vkm.", kind, spread[kind].sum())


if __name__ == "__main__":
    logging.basicConfig(filename=snakemake.log[0], level=logging.INFO, format="%(levelname)s %(message)s")  # noqa: F821
    epochs = snakemake.params.epochs  # noqa: F821
    versions = snakemake.params.versions  # noqa: F821
    main(
        edges_path=snakemake.input.edges,  # noqa: F821
        nodes_path=snakemake.input.nodes,  # noqa: F821
        legs_path=snakemake.input.legs,  # noqa: F821
        grid_path=snakemake.input.grid,  # noqa: F821
        polygons_path=snakemake.input.polygons,  # noqa: F821
        population_paths={e: snakemake.input[f"weight_{e}"] for e in epochs},  # noqa: F821
        nuts_rasters={v: snakemake.input[f"nuts_{v}"] for v in versions},  # noqa: F821
        nuts_codes_path=snakemake.input.nuts_codes,  # noqa: F821
        tonnes_path=snakemake.input.tonnes,  # noqa: F821
        codes=snakemake.params.codes,  # noqa: F821
        reference_year=snakemake.params.reference_year,  # noqa: F821
        ferry_sea_share=snakemake.params.ferry_sea_share,  # noqa: F821
        output_paths=dict(snakemake.output.items()),  # noqa: F821
    )
