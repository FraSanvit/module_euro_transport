"""ETISplus-based truck flows (Speth et al. 2022): domestic/international split and network vkm.

Per country, MULTIYEAR_PLAN.md §4.2, §6.1.

Facts used [verified in exploration]: edge flows are both directions combined; the
edges file has 2 exact duplicate rows (dropped); edge ID 2616216 is shared by two
different manually added links (OD flows are assigned to both).
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString

LOGGER = logging.getLogger(__name__)
ETIS_TO_EUROSTAT = {"GR": "EL"}
"""ETISplus country codes that differ from Eurostat/GISCO codes (UK is UK in both)."""


def read_edges(path) -> pd.DataFrame:
    """Edges without exact duplicate rows."""
    edges = pd.read_csv(path)
    edges = edges.drop(columns=[c for c in edges.columns if c.startswith("Unnamed")])
    before = len(edges)
    edges = edges.drop_duplicates().reset_index(drop=True)
    LOGGER.info("Edges: %d rows, %d exact duplicates dropped.", before, before - len(edges))
    return edges


def split_flows(od_chunks, zone_country: pd.Series, edges: pd.DataFrame, flow_column: str) -> pd.DataFrame:
    """Add each OD flow to every edge of its path, split domestic / international.

    `od_chunks` iterates DataFrames with `ID_origin_region, ID_destination_region,
    Edge_path_E_road, <flow_column>`; `zone_country` maps zone ID -> country. Returns
    `edges` with `flow_dom`, `flow_int` [trucks/year, both directions].
    """
    edge_ids = edges["Network_Edge_ID"].to_numpy()
    position = pd.Series(np.arange(len(edges)), index=edge_ids)
    position = position[~position.index.duplicated(keep="first")]
    flow = {"dom": np.zeros(len(edges)), "int": np.zeros(len(edges))}
    unknown = 0
    for chunk in od_chunks:
        chunk = chunk[chunk[flow_column] > 0]
        origin = zone_country.reindex(chunk["ID_origin_region"]).to_numpy()
        destination = zone_country.reindex(chunk["ID_destination_region"]).to_numpy()
        domestic = origin == destination
        paths = chunk["Edge_path_E_road"].astype(str).str.strip("[] ")
        lengths = paths.str.count(",").to_numpy() + 1
        lengths[paths.str.len().to_numpy() == 0] = 0
        joined = ",".join(paths[paths.str.len() > 0])
        ids = np.array(joined.split(","), dtype=np.int64) if joined else np.array([], dtype=np.int64)
        pos = position.reindex(ids).to_numpy()
        weight = np.repeat(chunk[flow_column].to_numpy(), lengths)
        is_dom = np.repeat(domestic, lengths)
        ok = ~np.isnan(pos)
        unknown += int((~ok).sum())
        pos = pos[ok].astype(int)
        np.add.at(flow["dom"], pos[is_dom[ok]], weight[ok][is_dom[ok]])
        np.add.at(flow["int"], pos[~is_dom[ok]], weight[ok][~is_dom[ok]])
    if unknown:
        LOGGER.warning("%d path entries reference unknown edge IDs (ignored).", unknown)
    for shared in np.unique(edge_ids[pd.Series(edge_ids).duplicated().to_numpy()]):
        rows = np.flatnonzero(edge_ids == shared)
        LOGGER.info("Edge ID %s is shared by %d links: OD flows assigned to all.", shared, len(rows))
        for kind in flow:
            flow[kind][rows[1:]] = flow[kind][rows[0]]
    out = edges.copy()
    out["flow_dom"] = flow["dom"]
    out["flow_int"] = flow["int"]
    return out


def check_split(edges: pd.DataFrame, flow_column: str) -> float:
    """Ratio of rebuilt (domestic + international) to published edge flow; logged."""
    ratio = (edges["flow_dom"] + edges["flow_int"]).sum() / edges[flow_column].sum()
    LOGGER.info("Rebuilt / published edge flow: %.4f", ratio)
    return ratio


def network_vkm_by_country(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    countries: gpd.GeoDataFrame,
    crs: str,
    ferry_sea_share: float,
    sea_handling: str = "drop",
) -> pd.DataFrame:
    """Network vkm per country, domestic and international OD flows (§6.1 step 3).

    Each edge is a straight line between its nodes. Its vkm (flow x `Distance`) is
    shared by the length of the line inside each country polygon (`countries` with
    column `CNTR_ID`). A manually added link with more than `ferry_sea_share` of its
    length on sea counts as sea over its whole length. The part on sea is dropped
    (`sea_handling="drop"`: trucks on a ferry drive no road km and use no fuel) or goes
    half to the country of each end node (`"ports"`, nodes file). Zero-length edges go
    to their end node's country. Returns `geo, etis_dom, etis_int` [vkm].
    """
    if sea_handling not in ("drop", "ports"):
        raise ValueError(f"sea_handling must be 'drop' or 'ports', not {sea_handling!r}.")
    xy = nodes.set_index("Network_Node_ID")[["Network_Node_X", "Network_Node_Y"]]
    node_country = nodes.set_index("Network_Node_ID")["Country"].replace(ETIS_TO_EUROSTAT)
    a = xy.reindex(edges["Network_Node_A_ID"]).to_numpy()
    b = xy.reindex(edges["Network_Node_B_ID"]).to_numpy()
    lines = gpd.GeoDataFrame(
        {"edge": np.arange(len(edges))},
        geometry=[LineString([p, q]) for p, q in zip(a, b)],
        crs="EPSG:4326",
    ).to_crs(crs)
    lines["length"] = lines.length
    polys = countries[["CNTR_ID", "geometry"]].to_crs(crs)
    pieces = gpd.overlay(lines, polys, how="intersection", keep_geom_type=True)
    pieces["share"] = pieces.length / pieces["length"].where(pieces["length"] > 0)
    land = pieces.groupby(["edge", "CNTR_ID"], as_index=False)["share"].sum()
    land_total = land.groupby("edge")["share"].sum().reindex(lines["edge"]).fillna(0.0)
    sea = (1 - land_total).clip(lower=0).to_numpy()
    ferry = (edges["Manually_Added"].to_numpy() == 1) & (sea > ferry_sea_share)
    LOGGER.info("Ferry rule: %d manually added links counted as sea.", int(ferry.sum()))
    zero_length = (lines["length"].to_numpy() == 0)
    sea = np.where(ferry, 1.0, np.where(zero_length, 0.0, sea))
    land = land[~land["edge"].isin(np.flatnonzero(ferry | zero_length))]
    to_ends = np.where(zero_length, 1.0, sea if sea_handling == "ports" else 0.0)
    vkm = {k: edges[f"flow_{k}"].to_numpy() * edges["Distance"].to_numpy() for k in ("dom", "int")}
    parts = [
        pd.DataFrame(
            {
                "geo": land["CNTR_ID"].to_numpy(),
                "etis_dom": vkm["dom"][land["edge"]] * land["share"].to_numpy(),
                "etis_int": vkm["int"][land["edge"]] * land["share"].to_numpy(),
            }
        )
    ]
    for end in ("Network_Node_A_ID", "Network_Node_B_ID"):
        parts.append(
            pd.DataFrame(
                {
                    "geo": node_country.reindex(edges[end]).to_numpy(),
                    "etis_dom": vkm["dom"] * to_ends / 2,
                    "etis_int": vkm["int"] * to_ends / 2,
                }
            )
        )
    out = pd.concat(parts, ignore_index=True).groupby("geo", as_index=False).sum()
    total = vkm["dom"].sum() + vkm["int"].sum()
    LOGGER.info(
        "Network vkm: %.4g total, %.4g attributed (%.1f%% on sea, %s).",
        total, out[["etis_dom", "etis_int"]].to_numpy().sum(),
        100 * (sea * (vkm["dom"] + vkm["int"])).sum() / total,
        "dropped" if sea_handling == "drop" else "split to end-node countries",
    )  # fmt: skip
    return out
