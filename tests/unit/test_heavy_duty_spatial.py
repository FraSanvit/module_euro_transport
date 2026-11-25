"""Unit tests for the heavy-duty spatial allocation (MULTIYEAR_PLAN.md §6.2, §6.10)."""

import geopandas as gpd
import numpy as np
import pytest
from _raster import NODATA_ID, grid_from_bounds, rasterize_ids, read_raster, write_raster
from prepare_heavy_duty_grid import carve, scope_polygons
from shapely.geometry import box

CRS = "EPSG:3035"


def test_grid_from_bounds_snaps_and_buffers():
    transform, shape = grid_from_bounds((1050.0, 2020.0, 3900.0, 4010.0), 1000, buffer=500)
    assert (transform.c, transform.f) == (0.0, 5000.0)  # 550 -> 0, 4510 -> 5000
    assert shape == (4, 5)  # y 1000..5000, x 0..5000


def test_rasterize_ids_uses_cell_centres(tmp_path):
    transform, shape = grid_from_bounds((0, 0, 4, 2), 1)
    # Polygon A covers x 0..1.6 (centres 0.5, 1.5); B covers x 1.6..4 (centres 2.5, 3.5).
    ids = rasterize_ids([box(0, 0, 1.6, 2), box(1.6, 0, 4, 2)], transform, shape)
    assert ids.tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    none = rasterize_ids([box(10, 10, 11, 11)], transform, shape)
    assert (none == NODATA_ID).all()
    write_raster(tmp_path / "ids.tif", ids.astype("int16"), transform, CRS, nodata=NODATA_ID)
    back, profile = read_raster(tmp_path / "ids.tif")
    assert (back == ids).all() and profile["crs"].to_string() == CRS


def test_scope_polygons_clips_to_box_and_maps_codes():
    lonlat = gpd.GeoDataFrame(
        {"CNTR_ID": ["AA", "EL", "ZZ"]},
        geometry=[box(0, 40, 2, 42).union(box(0, 20, 2, 22)), box(20, 38, 22, 40), box(5, 40, 6, 41)],
        crs="EPSG:4326",
    )
    out = scope_polygons(lonlat, {"AAA": "AA", "GRC": "EL"}, ["AAA", "GRC"], CRS, (-25, 33, 45, 72))
    assert out["country_id"].tolist() == ["AAA", "GRC"]
    aaa = out.set_index("country_id").to_crs("EPSG:4326").loc["AAA", "geometry"]
    assert aaa.bounds[1] == pytest.approx(40, abs=1e-6)  # the part at 20-22 N is dropped
    with pytest.raises(ValueError, match="No GISCO polygon"):
        scope_polygons(lonlat, {"BBB": "BB"}, ["BBB"], CRS, (-25, 33, 45, 72))
    assert np.isfinite(out.area).all()


def test_carve_cuts_a_country_out_of_its_parent():
    countries = gpd.GeoDataFrame({"CNTR_ID": ["RS"]}, geometry=[box(0, 0, 10, 10)], crs=CRS)
    kosovo = gpd.GeoDataFrame(geometry=[box(6, 0, 12, 4)], crs=CRS)  # partly outside the parent
    out = carve(countries, {"SRB": "RS", "XKX": "XK"}, {"XKX": ("SRB", kosovo)}, CRS).set_index("CNTR_ID")
    assert out.loc["XK", "geometry"].area == pytest.approx(16)  # only the part inside RS
    assert out.loc["RS", "geometry"].area == pytest.approx(84)


def test_tonnes_table_and_data_year():
    import pandas as pd
    from _short_haul import data_year, tonnes_table

    loaded = pd.DataFrame({"geo": ["AA001", "AA002", "AA", "AAZZZ"], "year": 2019, "value": [10.0, 4, 99, 7]})
    unloaded = pd.DataFrame({"geo": ["AA001"], "year": 2019, "value": [20.0]})
    out = tonnes_table(loaded, unloaded).set_index("nuts")["tonnes"]
    assert out.to_dict() == {"AA001": 15_000.0, "AA002": 4_000.0}  # national and extra-regio dropped
    assert data_year([2016, 2018, 2020], 2019) == 2018
    assert data_year([2016, 2018, 2020], 2010) == 2016
    assert data_year([], 2019) is None


def test_short_haul_key_tonnes_population_and_fallbacks():
    import pandas as pd
    from _short_haul import short_haul_key

    # Country AAA: 4 cells, regions R1 (cells 0-1) and R2 (cell 2); cell 3 in no region.
    # Country BBB: 2 cells, no tonnes -> population.
    population = np.array([[1.0, 3.0, 2.0, 4.0, 5.0, 5.0]])
    countries_ids = np.array([[0, 0, 0, 0, 1, 1]])
    regions = {"2016": np.array([[0, 0, 1, -1, -1, -1]])}
    codes = {"2016": ["AA001", "AA002"]}
    tonnes = pd.DataFrame({"nuts": ["AA001", "AA002", "AA001"], "year": [2018, 2018, 2015], "tonnes": [40.0, 20.0, 1.0]})
    key, report = short_haul_key(population, countries_ids, ["AAA", "BBB"], {"AAA": "AA", "BBB": "BB"},
                                 tonnes, regions, codes, 2019)  # fmt: skip
    # R1: q = 40 / 4 = 10 -> cells 10, 30; R2: q = 20 / 2 = 10 -> 20; cell 3: country mean q 10 -> 40.
    assert key[0, :4].tolist() == pytest.approx([10, 30, 20, 40])
    assert key[0, 4:].tolist() == [5, 5]  # BBB: population
    report = report.set_index("country_id")
    assert report.loc["AAA", "data_year"] == 2018 and report.loc["BBB", "method"] == "population"


def test_edge_pieces_and_rasterise_network():
    import pandas as pd
    from prepare_heavy_duty_etisplus_keys import edge_pieces, rasterise_network
    from pyproj import Transformer

    px, py, edge, n = edge_pieces(np.array([0.0]), np.array([0.0]), np.array([10.0]), np.array([0.0]), 2.5)
    assert n.tolist() == [4] and px.tolist() == [1.25, 3.75, 6.25, 8.75] and (edge == 0).all()
    # Grid 4 x 1 km cells; land in the first three. Edge 1: road over cells 0-1 (4 km road distance).
    # Edge 2: manually added, three quarters over the sea cell -> ferry, dropped.
    transform, shape = grid_from_bounds((4_000_000, 3_000_000, 4_004_000, 3_001_000), 1000)
    land = np.array([[True, True, True, False]])
    to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    xs = [4_000_100, 4_001_900, 4_002_600, 4_003_900]
    lon, lat = to_lonlat.transform(xs, [3_000_500] * 4)
    nodes = pd.DataFrame({"Network_Node_ID": [1, 2, 3, 4], "Network_Node_X": lon, "Network_Node_Y": lat})
    edges = pd.DataFrame({"Network_Node_A_ID": [1, 3], "Network_Node_B_ID": [2, 4], "Manually_Added": [0, 1],
                          "Distance": [4.0, 2.0], "flow_dom": [10.0, 5.0], "flow_int": [1.0, 1.0]})  # fmt: skip
    out, ferries = rasterise_network(edges, nodes, land, transform, CRS, 0.5)
    assert ferries == 1
    assert out["dom"].sum() == pytest.approx(40) and out["dom"][0, 2:].sum() == 0
    assert out["dom"][0, 0] == pytest.approx(out["dom"][0, 1], rel=0.2)


def test_spread_legs_by_key_within_region():
    import pandas as pd
    from prepare_heavy_duty_etisplus_keys import spread_legs

    key = np.array([[1.0, 3.0, 0.0, 0.0, 2.0]])
    country_ids = np.array([[0, 0, 0, 0, 1]])
    regions = {"2013": np.array([[0, 0, 1, 1, -1]])}
    legs = pd.DataFrame({"geo": ["AAA", "AAA", "BBB"], "legs_dom": [8.0, 4.0, 6.0], "legs_int": [0.0, 0.0, 2.0]})
    out = spread_legs(legs, [("2013", 0), ("2013", 1), ("country", 1)], key, country_ids, ["AAA", "BBB"], regions)
    assert out["dom"][0].tolist() == pytest.approx([2, 6, 2, 2, 6])  # region 1 has zero key: evenly
    assert out["int"][0].tolist() == pytest.approx([0, 0, 0, 0, 2])  # zone without region: its country
    dropped = spread_legs(legs, [("2013", 0), ("2013", 1), None], key, country_ids, ["AAA", "BBB"], regions)
    assert dropped["int"].sum() == 0  # zone outside the grid: left out of the key


def test_population_for_year(tmp_path):
    from prepare_heavy_duty_etisplus_keys import population_for_year

    transform, shape = grid_from_bounds((0, 0, 2, 1), 1)
    paths = {}
    for epoch, value in [(2015, 10.0), (2020, 20.0)]:
        paths[epoch] = tmp_path / f"p{epoch}.tif"
        write_raster(paths[epoch], np.full(shape, value, dtype="float32"), transform, CRS)
    assert population_for_year(paths, 2018)[0, 0] == pytest.approx(16)
    assert population_for_year(paths, 2010)[0, 0] == pytest.approx(10)
    assert population_for_year(paths, 2024)[0, 0] == pytest.approx(20)


def test_allocate_with_gregor_conserves_levels_and_falls_back():
    from allocate_heavy_duty_vkm import allocate, choose_keys, combined_key

    transform, shape = grid_from_bounds((0, 0, 5, 1), 1)
    profile = {"transform": transform, "height": shape[0], "width": shape[1], "crs": CRS}
    polygons = gpd.GeoDataFrame({"country_id": ["AAA", "BBB"]}, geometry=[box(0, 0, 2, 1), box(2, 0, 4, 1)], crs=CRS)
    ids = rasterize_ids(polygons.geometry, transform, shape)
    assert ids.tolist() == [[0, 0, 1, 1, -1]]
    primary = np.array([[1.0, 3.0, 0.0, 0.0, 9.0]])  # BBB: zero primary key
    secondary = np.array([[1.0, 1.0, 2.0, 6.0, 9.0]])
    used = choose_keys([primary, secondary], [[0, 1], [0, 1]], ids)
    assert used.tolist() == [0, 1]
    key = combined_key([primary, secondary], used, ids)
    assert key[0].tolist() == [1, 3, 2, 6, 0]
    out = allocate(polygons, np.array([8.0, 4.0]), key, profile)
    assert out[0].tolist() == pytest.approx([2, 6, 1, 3, 0])  # outside any country: 0
    assert choose_keys([primary, secondary], [[1], [1]], ids).tolist() == [1, 1]  # order per country


def test_aggregate_to_shapes_with_diagnostics(tmp_path):
    import pandas as pd
    from aggregate_heavy_duty_vkm import aggregate, membership

    transform, shape = grid_from_bounds((0, 0, 4, 1), 1)
    profile = {"transform": transform, "height": shape[0], "width": shape[1], "crs": CRS}
    shapes = gpd.GeoDataFrame({"shape_id": ["S1", "S2"], "country_id": ["AAA", "BBB"]},
                              geometry=[box(0, 0, 1.4, 1), box(1.4, 0, 3, 1)], crs=CRS)  # fmt: skip
    shape_ids = membership(shapes, profile)
    assert shape_ids.tolist() == [[0, 1, 1, -1]]
    path = tmp_path / "v.tif"
    write_raster(path, np.array([[1.0, 2.0, 4.0, 8.0]], dtype="float32"), transform, CRS)
    country_ids = np.array([[0, 0, 1, 1]], dtype="int32")
    table, diag = aggregate(shapes, shape_ids, country_ids, ["AAA", "BBB"], {(2019, "international"): path})
    assert table.set_index("shape_id")["vkm"].to_dict() == pytest.approx({"S1": 1, "S2": 6})
    d = diag.set_index("country_id")
    assert d.loc["AAA", "vkm_in_other_country_shapes"] == pytest.approx(2)  # cell 1 of AAA in S2 (BBB)
    assert d.loc["BBB", "vkm_outside_shapes"] == pytest.approx(8)
    assert isinstance(table, pd.DataFrame) and set(table["vehicle_category"]) == {"heavy_duty"}
