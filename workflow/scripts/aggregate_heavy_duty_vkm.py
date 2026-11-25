"""Heavy-duty vkm per user shape from the vkm rasters (MULTIYEAR_PLAN.md §6.10).

Each cell belongs to the shape containing its centre (gregor membership matrix, as in
the disaggregation). Writes vkm per shape, year and component, and per country and year
the vkm outside any shape and in shapes of another country (diagnostics).
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 (registers the .rio accessor)
import xarray as xr
from _raster import cell_centres, read_raster
from gregor.disaggregate import get_belongs_to_matrix

LOGGER = logging.getLogger(__name__)
COMPONENTS = ["international", "domestic_long", "domestic_short"]


def membership(shapes: gpd.GeoDataFrame, profile) -> np.ndarray:
    """Index of the shape containing each cell centre (-1 outside), via gregor."""
    x, y = cell_centres(profile["transform"], (profile["height"], profile["width"]))
    template = xr.DataArray(np.zeros((len(y), len(x)), dtype="float32"), coords={"y": y, "x": x}, dims=("y", "x"))
    template = template.rio.write_crs(profile["crs"]).rio.write_transform(profile["transform"])
    return get_belongs_to_matrix(template, shapes.to_crs(profile["crs"]).geometry, nodata=-1).to_numpy()


def aggregate(shapes, shape_ids, country_ids, countries, rasters: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per shape and per country sums for each (year, component) raster in `rasters`."""
    rows, diagnostics = [], []
    in_shape = shape_ids >= 0
    in_country = country_ids >= 0
    shape_country = shapes["country_id"].to_numpy()
    for (year, component), path in sorted(rasters.items()):
        values = read_raster(path)[0].astype("float64")
        per_shape = np.bincount(shape_ids[in_shape], weights=values[in_shape], minlength=len(shapes))
        rows.append(pd.DataFrame({"shape_id": shapes["shape_id"], "country_id": shape_country, "year": year,
                                  "vehicle_category": "heavy_duty", "component": component, "vkm": per_shape}))  # fmt: skip
        total = np.bincount(country_ids[in_country], weights=values[in_country], minlength=len(countries))
        outside_mask = in_country & ~in_shape
        outside = np.bincount(country_ids[outside_mask], weights=values[outside_mask], minlength=len(countries))
        both = in_country & in_shape
        cell_country = np.array(countries, dtype=object)[country_ids[both]]
        other = cell_country != shape_country[shape_ids[both]]
        elsewhere = np.bincount(country_ids[both][other], weights=values[both][other], minlength=len(countries))
        diagnostics.append(pd.DataFrame({"country_id": countries, "year": year, "component": component, "vkm": total,
                                         "vkm_outside_shapes": outside, "vkm_in_other_country_shapes": elsewhere}))  # fmt: skip
    return pd.concat(rows, ignore_index=True), pd.concat(diagnostics, ignore_index=True)


def main(shapes_path, grid_path, polygons_path, rasters: dict, output_path, diagnostics_path):
    """Aggregate the rasters to the user shapes and write the tables."""
    shapes = gpd.read_parquet(shapes_path).reset_index(drop=True)
    for column in ["shape_id", "country_id"]:
        if column not in shapes:
            raise ValueError(f"User shapes need a `{column}` column.")
    country_ids, profile = read_raster(grid_path)
    countries = gpd.read_parquet(polygons_path)["country_id"].tolist()
    shape_ids = membership(shapes, profile)
    empty = shapes.loc[np.bincount(shape_ids[shape_ids >= 0], minlength=len(shapes)) == 0, "shape_id"].tolist()
    if empty:
        LOGGER.warning("Shapes without any cell centre (vkm 0): %s.", empty)
    table, diagnostics = aggregate(shapes, shape_ids, country_ids.astype("int32"), countries, rasters)
    for row in diagnostics.groupby(["country_id"])[["vkm", "vkm_outside_shapes"]].sum().itertuples():
        if row.vkm > 0 and row.vkm_outside_shapes > 1e-3 * row.vkm:
            LOGGER.warning("%s: %.2f%% of heavy-duty vkm outside the user shapes.", row.Index, 100 * row.vkm_outside_shapes / row.vkm)
    table.to_parquet(output_path, index=False)
    diagnostics.to_parquet(diagnostics_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(filename=snakemake.log[0], level=logging.INFO, format="%(levelname)s %(message)s")  # noqa: F821
    years = snakemake.params.years  # noqa: F821
    main(
        shapes_path=snakemake.input.shapes,  # noqa: F821
        grid_path=snakemake.input.grid,  # noqa: F821
        polygons_path=snakemake.input.polygons,  # noqa: F821
        rasters={(y, c): snakemake.input[f"{c}_{y}"] for y in years for c in COMPONENTS},  # noqa: F821
        output_path=snakemake.output.vkm,  # noqa: F821
        diagnostics_path=snakemake.output.diagnostics,  # noqa: F821
    )
