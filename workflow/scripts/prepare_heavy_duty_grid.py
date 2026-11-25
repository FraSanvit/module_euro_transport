"""Grid and country raster for the heavy-duty spatial allocation (MULTIYEAR_PLAN.md §6.2).

Country polygons (GISCO 1:3 million, EPSG:3035) of the processed countries, clipped to
the European bounding box (overseas territories, the Canaries and Madeira fall outside;
their share stays with the rest of the country). The grid covers the polygons plus a
buffer, snapped to the resolution. Each cell gets the index of the country containing
its centre.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from _heavy_duty import verify_sha256
from _raster import NODATA_ID, grid_from_bounds, rasterize_ids, write_raster
from shapely.geometry import box

LOGGER = logging.getLogger(__name__)


def carve(countries: gpd.GeoDataFrame, codes: dict, carved: dict, crs: str) -> gpd.GeoDataFrame:
    """Add countries GISCO does not separate, cut out of their parent's polygon.

    `carved` maps ISO3 -> (parent ISO3, GeoDataFrame of the country). The country becomes
    its polygon inside the parent; the parent loses that part.
    """
    countries = countries.to_crs(crs).copy()
    for iso, (parent, shape) in carved.items():
        hit = countries["CNTR_ID"] == codes[parent]
        outline = countries.loc[hit].geometry.union_all()
        piece = shape.to_crs(crs).geometry.union_all().intersection(outline)
        countries.loc[hit, "geometry"] = countries.loc[hit].geometry.difference(piece)
        row = gpd.GeoDataFrame({"CNTR_ID": [codes[iso]]}, geometry=[piece], crs=crs)
        countries = pd.concat([countries, row], ignore_index=True)
        LOGGER.info("%s cut out of %s: %.0f km2.", iso, parent, piece.area / 1e6)
    return gpd.GeoDataFrame(countries, geometry="geometry", crs=crs)


def scope_polygons(countries: gpd.GeoDataFrame, codes: dict, scope: list[str], crs: str, bbox_lonlat) -> gpd.GeoDataFrame:
    """Polygons of `scope` (ISO3), clipped to `bbox_lonlat`, in `crs`: `country_id, geometry`."""
    to_iso = {codes[c]: c for c in scope if c in codes}
    selected = countries[countries["CNTR_ID"].isin(to_iso)].to_crs(crs)
    missing = sorted(set(to_iso.values()) - set(selected["CNTR_ID"].map(to_iso)))
    if missing:
        raise ValueError(f"No GISCO polygon for {missing}.")
    frame = gpd.GeoDataFrame(geometry=[box(*bbox_lonlat).segmentize(0.1)], crs="EPSG:4326").to_crs(crs)
    clipped = selected.clip(frame.geometry.iloc[0])
    for cntr, before, after in zip(selected["CNTR_ID"], selected.area, clipped.reindex(selected.index).area):
        if after < before * (1 - 1e-6):
            LOGGER.info("%s: %.0f km2 outside the European bounding box dropped.", to_iso[cntr], (before - after) / 1e6)
    out = gpd.GeoDataFrame({"country_id": clipped["CNTR_ID"].map(to_iso).to_numpy()}, geometry=clipped.geometry.to_numpy(), crs=crs)
    return out.sort_values("country_id", ignore_index=True)


def main(
    countries_path, codes, scope, crs, resolution, bbox_lonlat, buffer_km, raster_path, polygons_path,
    carved_paths=None, carved=None,
):  # fmt: skip
    """Write the country raster and the polygons it was built from."""
    countries = gpd.read_file(countries_path)
    shapes = {}
    for iso, path in (carved_paths or {}).items():
        verify_sha256(path, carved[iso]["sha256"])
        shapes[iso] = (carved[iso]["parent"], gpd.read_file(path))
    if shapes:
        countries = carve(countries, codes, shapes, crs)
    polygons = scope_polygons(countries, codes, scope, crs, bbox_lonlat)
    transform, shape = grid_from_bounds(polygons.total_bounds, resolution, buffer_km * 1000)
    ids = rasterize_ids(polygons.geometry, transform, shape)
    LOGGER.info("Grid %s, %d x %d cells of %g m, origin (%.0f, %.0f).", crs, shape[0], shape[1], resolution, transform.c, transform.f)
    counts = np.bincount(ids[ids != NODATA_ID], minlength=len(polygons))
    for country, n in zip(polygons["country_id"], counts):
        LOGGER.info("%s: %d cells.", country, n)
    empty = [c for c, n in zip(polygons["country_id"], counts) if n == 0]
    if empty:
        raise ValueError(f"Countries without any cell: {empty}.")
    write_raster(raster_path, ids.astype("int16"), transform, crs, nodata=NODATA_ID)
    polygons.to_parquet(polygons_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(filename=snakemake.log[0], level=logging.INFO, format="%(levelname)s %(message)s")  # noqa: F821
    main(
        countries_path=snakemake.input.countries,  # noqa: F821
        codes=snakemake.params.codes,  # noqa: F821
        scope=snakemake.params.scope,  # noqa: F821
        crs=snakemake.params.crs,  # noqa: F821
        resolution=snakemake.params.resolution,  # noqa: F821
        bbox_lonlat=snakemake.params.bbox_lonlat,  # noqa: F821
        buffer_km=snakemake.params.buffer_km,  # noqa: F821
        raster_path=snakemake.output.raster,  # noqa: F821
        polygons_path=snakemake.output.polygons,  # noqa: F821
        carved_paths={k: v for k, v in snakemake.input.items() if k != "countries"},  # noqa: F821
        carved=snakemake.params.carved,  # noqa: F821
    )
