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


LEG_COLUMNS = ("Distance_from_origin_region_to_E_road", "Distance_from_E_road_to_destination_region")
"""OD columns of the first leg (origin zone -> E-road network) and last leg (E-road -> destination zone)."""


def zone_legs(chunk: pd.DataFrame, zone_country: pd.Series, flow_column: str) -> pd.DataFrame:
    """First and last legs of OD flows per zone, domestic / international (§6.1 step 4).

    The first leg is driven in the origin zone, the last leg in the destination zone;
    neither is on an E-road edge. Returns `zone, kind (dom | int), legs` [vkm] for the chunk.
    """
    chunk = chunk[chunk[flow_column] > 0]
    origin = zone_country.reindex(chunk["ID_origin_region"]).to_numpy()
    destination = zone_country.reindex(chunk["ID_destination_region"]).to_numpy()
    kind = np.where(origin == destination, "dom", "int")
    flow = chunk[flow_column].to_numpy()
    parts = pd.DataFrame(
        {
            "zone": np.concatenate([chunk["ID_origin_region"].to_numpy(), chunk["ID_destination_region"].to_numpy()]),
            "kind": np.concatenate([kind, kind]),
            "legs": np.concatenate([flow * chunk[LEG_COLUMNS[0]].to_numpy(), flow * chunk[LEG_COLUMNS[1]].to_numpy()]),
        }
    )
    return parts.groupby(["zone", "kind"], as_index=False)["legs"].sum()


def legs_by_zone(parts, regions: pd.DataFrame) -> pd.DataFrame:
    """Sum the `zone_legs` results per zone.

    One row per zone with `zone, geo` (Eurostat code), the zone centre `x, y` (lon/lat)
    and `legs_dom, legs_int` [vkm].
    """
    legs = pd.concat(parts).groupby(["zone", "kind"])["legs"].sum().unstack("kind")
    legs = legs.reindex(columns=["dom", "int"]).fillna(0.0)
    zones = regions.set_index("ETISPlus_Zone_ID")
    out = pd.DataFrame(
        {
            "zone": legs.index.to_numpy(),
            "geo": zones["Country"].reindex(legs.index).replace(ETIS_TO_EUROSTAT).to_numpy(),
            "x": zones["Geometric_center_X"].reindex(legs.index).to_numpy(),
            "y": zones["Geometric_center_Y"].reindex(legs.index).to_numpy(),
            "legs_dom": legs["dom"].to_numpy(),
            "legs_int": legs["int"].to_numpy(),
        }
    )
    LOGGER.info(
        "First and last legs: %.4g vkm domestic, %.4g vkm international, %d zones.",
        out["legs_dom"].sum(), out["legs_int"].sum(), len(out),
    )  # fmt: skip
    return out


def add_zone_legs(network: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    """Add the first and last legs of each country's zones: `etis_dom_legs, etis_int_legs` [vkm]."""
    per_country = legs.groupby("geo")[["legs_dom", "legs_int"]].sum()
    per_country.columns = ["etis_dom_legs", "etis_int_legs"]
    out = network.set_index("geo").join(per_country, how="left")
    out[["etis_dom_legs", "etis_int_legs"]] = out[["etis_dom_legs", "etis_int_legs"]].fillna(0.0)
    return out.reset_index()


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
    length on sea is a ferry, on sea over its whole length.

    `sea_handling="drop"`: ferry km are dropped (trucks on a ferry drive no road km and
    use no fuel). Other edges are roads even where the straight line crosses water
    (coastal roads across bays, bridges such as Great Belt or Oresund): their full vkm
    is kept, the land shares scaled up to 1, or, with no land at all, half to the
    country of each end node. `"ports"`: every sea part goes half to the country of
    each end node (nodes file). Zero-length edges go to their end node's country.
    Returns `geo, etis_dom, etis_int` [vkm].
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
    land = land[~land["edge"].isin(np.flatnonzero(ferry | zero_length))].copy()
    if sea_handling == "ports":
        to_ends = np.where(zero_length, 1.0, sea)
    else:
        # Roads over water keep their vkm: land shares scaled to 1; all-sea roads to the ends.
        has_land = land_total.to_numpy() > 0
        land["share"] = land["share"] / land_total.to_numpy()[land["edge"]]
        road_sea = ~ferry & (sea > 0)
        to_ends = np.where(zero_length | (road_sea & ~has_land), 1.0, 0.0)
        LOGGER.info(
            "Roads over water kept: %.4g vkm (%d edges); ferries dropped: %.4g vkm (%d links).",
            (sea * road_sea * (edges["flow_dom"] + edges["flow_int"]) * edges["Distance"]).sum(), int(road_sea.sum()),
            (ferry * (edges["flow_dom"] + edges["flow_int"]) * edges["Distance"]).sum(), int(ferry.sum()),
        )  # fmt: skip
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
        "Network vkm: %.4g total, %.4g attributed (%.1f%% on sea; %s).",
        total, out[["etis_dom", "etis_int"]].to_numpy().sum(),
        100 * (sea * (vkm["dom"] + vkm["int"])).sum() / total,
        "ferries dropped, roads over water kept" if sea_handling == "drop" else "split to end-node countries",
    )  # fmt: skip
    return out
