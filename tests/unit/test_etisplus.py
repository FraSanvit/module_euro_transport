"""Unit tests for the ETISplus helpers and the heavy-duty settings."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import yaml
from _etisplus import add_zone_legs, legs_by_zone, network_vkm_by_country, split_flows, zone_legs
from shapely.geometry import box

MODULE_ROOT = Path(__file__).parents[2]


def edges_table():
    return pd.DataFrame(
        {
            "Network_Edge_ID": [1, 2, 3, 3],  # ID 3 shared by two links
            "Manually_Added": [0, 0, 1, 1],
            "Distance": [10.0, 20.0, 5.0, 5.0],
            "Network_Node_A_ID": [10, 11, 12, 12],
            "Network_Node_B_ID": [11, 12, 13, 13],
        }
    )


def test_split_flows_domestic_international_and_shared_ids():
    zones = pd.Series({100: "AA", 101: "AA", 200: "BB"})
    od = pd.DataFrame(
        {
            "ID_origin_region": [100, 100, 100],
            "ID_destination_region": [101, 200, 200],
            "Edge_path_E_road": ["[1, 2]", "[2, 3]", "[]"],
            "flow": [5.0, 7.0, 9.0],
        }
    )
    out = split_flows([od], zones, edges_table(), "flow")
    assert out["flow_dom"].tolist() == [5, 5, 0, 0]
    assert out["flow_int"].tolist() == [0, 7, 7, 7]  # shared ID 3 assigned to both links


def test_network_vkm_by_country_land_and_sea():
    """Two square countries side by side; an edge across the border splits by length;
    an edge over sea is dropped, or goes half to each end node's country ("ports")."""
    crs = "EPSG:3035"
    polys = gpd.GeoDataFrame({"CNTR_ID": ["AA", "BB"]}, geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=crs)
    nodes = gpd.GeoDataFrame(
        {"Network_Node_ID": [1, 2, 3, 4], "Country": ["AA", "BB", "AA", "GR"]},
        geometry=gpd.points_from_xy([50, 150, 50, 50], [50, 50, 50, 300]),
        crs=crs,
    ).to_crs("EPSG:4326")
    nodes = pd.DataFrame({"Network_Node_ID": nodes["Network_Node_ID"], "Country": nodes["Country"],
                          "Network_Node_X": nodes.geometry.x, "Network_Node_Y": nodes.geometry.y})  # fmt: skip
    edges = pd.DataFrame(
        {
            "Manually_Added": [0, 1],
            "Distance": [1.0, 1.0],
            "Network_Node_A_ID": [1, 3],
            "Network_Node_B_ID": [2, 4],
            "flow_dom": [100.0, 0.0],
            "flow_int": [0.0, 60.0],
        }
    )
    out = network_vkm_by_country(edges, nodes, polys, crs, ferry_sea_share=0.5, sea_handling="ports").set_index("geo")
    assert out.loc["AA", "etis_dom"] == pytest.approx(50, rel=1e-3)
    assert out.loc["BB", "etis_dom"] == pytest.approx(50, rel=1e-3)
    # Ferry (80% on sea) -> all sea -> half to AA, half to the end node in GR (Eurostat EL).
    assert out.loc["AA", "etis_int"] == pytest.approx(30)
    assert out.loc["EL", "etis_int"] == pytest.approx(30)
    # Default: sea km dropped (no road km on a ferry); land unchanged.
    drop = network_vkm_by_country(edges, nodes, polys, crs, ferry_sea_share=0.5).set_index("geo")
    assert drop.loc["AA", "etis_dom"] == pytest.approx(50, rel=1e-3)
    assert drop["etis_int"].sum() == pytest.approx(0, abs=1e-9)
    # Road edges (not manually added) crossing water keep their full vkm: half on land in
    # AA -> all to AA; a bridge fully over sea (AA node 5 to GR node 4) -> half to each end.
    roads = pd.DataFrame(
        {
            "Manually_Added": [0, 0],
            "Distance": [1.0, 1.0],
            "Network_Node_A_ID": [1, 5],
            "Network_Node_B_ID": [5, 4],
            "flow_dom": [40.0, 0.0],
            "flow_int": [0.0, 20.0],
        }
    )
    extra = gpd.GeoDataFrame({"Network_Node_ID": [5], "Country": ["AA"]}, geometry=gpd.points_from_xy([50], [150]), crs=crs).to_crs("EPSG:4326")
    nodes5 = pd.concat([nodes, pd.DataFrame({"Network_Node_ID": [5], "Country": ["AA"], "Network_Node_X": extra.geometry.x, "Network_Node_Y": extra.geometry.y})], ignore_index=True)  # fmt: skip
    kept = network_vkm_by_country(roads, nodes5, polys, crs, ferry_sea_share=0.5).set_index("geo")
    assert kept.loc["AA", "etis_dom"] == pytest.approx(40, rel=1e-3)
    assert kept.loc["AA", "etis_int"] == pytest.approx(10, rel=1e-3)
    assert kept.loc["EL", "etis_int"] == pytest.approx(10, rel=1e-3)
    with pytest.raises(ValueError, match="sea_handling"):
        network_vkm_by_country(edges, nodes, polys, crs, ferry_sea_share=0.5, sea_handling="nope")


def test_zone_legs_first_and_last_leg_per_zone():
    """First leg to the origin zone, last leg to the destination zone, by OD type."""
    zones = pd.Series({100: "AA", 101: "AA", 200: "GR"})
    od = pd.DataFrame(
        {
            "ID_origin_region": [100, 100, 101],
            "ID_destination_region": [101, 200, 100],
            "Distance_from_origin_region_to_E_road": [10.0, 4.0, 1.0],
            "Distance_from_E_road_to_destination_region": [5.0, 30.0, 2.0],
            "flow": [2.0, 3.0, 0.0],  # zero flow ignored
        }
    )
    regions = pd.DataFrame({"ETISPlus_Zone_ID": [100, 101, 200], "Country": ["AA", "AA", "GR"],
                            "Geometric_center_X": [1.0, 2, 3], "Geometric_center_Y": [4.0, 5, 6]})  # fmt: skip
    half = len(od) // 2
    legs = legs_by_zone([zone_legs(od.iloc[:half], zones, "flow"), zone_legs(od.iloc[half:], zones, "flow")], regions)
    legs = legs.set_index("zone")
    assert legs.loc[100, "legs_dom"] == pytest.approx(2 * 10)
    assert legs.loc[100, "legs_int"] == pytest.approx(3 * 4)
    assert legs.loc[101, "legs_dom"] == pytest.approx(2 * 5)
    assert legs.loc[200, "legs_int"] == pytest.approx(3 * 30)
    assert legs.loc[200, "geo"] == "EL"  # ETISplus GR -> Eurostat EL
    network = pd.DataFrame({"geo": ["AA", "BB"], "etis_dom": [100.0, 50], "etis_int": [40.0, 10]})
    out = add_zone_legs(network, legs.reset_index()).set_index("geo")
    assert out.loc["AA", "etis_dom_legs"] == pytest.approx(30)
    assert out.loc["AA", "etis_int_legs"] == pytest.approx(12)
    assert out.loc["BB", "etis_int_legs"] == 0  # no zones: no legs
    assert "EL" not in out.index  # only countries with a network row


def load(path):
    return yaml.safe_load((MODULE_ROOT / path).read_text(encoding="utf-8"))


def test_heavy_duty_settings_are_consistent():
    hd = load("workflow/internal/settings.yaml")["heavy_duty"]
    supported = set(load("workflow/internal/config.schema.yaml")["properties"]["countries"]["items"]["enum"])
    neighbours = load("workflow/internal/settings.yaml")["gap_filling"]["neighbours"]
    every = supported | {n for names in neighbours.values() for n in names}
    assert every <= set(hd["country_codes"])
    for key in ["cabotage_zero", "islands", "toll_countries"]:
        assert set(hd[key]) <= supported, key
    assert set(hd["exclude"]) <= supported
    for parts in hd["exclude"].values():
        assert set(parts) <= {"domestic", "own_international"}
    assert set(hd["territorial_totals"]["overrides"]) == {"GBR", "SWE", "CHE"}
    for file in hd["etisplus"]["files"].values():
        assert len(file["sha256"]) == 64
