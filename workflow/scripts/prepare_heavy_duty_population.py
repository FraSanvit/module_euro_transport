"""Population raster on the allocation grid for one GHS-POP epoch (MULTIYEAR_PLAN.md §6.2).

GHS-POP R2023A, 1 km, Mollweide (ESRI:54009). The part covering the grid is read from
the zip, nodata set to 0, and reprojected to the grid by summing (persons per cell).
Years between epochs are interpolated later (`population_for_year`).
"""

import logging
import zipfile

import geopandas as gpd
import numpy as np
import rasterio
from _raster import read_raster, write_raster
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds

LOGGER = logging.getLogger(__name__)
MARGIN = 20_000  # m, around the grid in the source projection


def tif_in_zip(zip_path) -> str:
    """GDAL path of the GeoTIFF inside a GHS-POP zip."""
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".tif")]
    if len(names) != 1:
        raise ValueError(f"{zip_path}: expected one GeoTIFF, found {names}.")
    return f"zip://{zip_path}!/{names[0]}"


def to_grid(source_path, grid_profile) -> np.ndarray:
    """Population of the source raster summed into the grid cells (float64)."""
    height, width = grid_profile["height"], grid_profile["width"]
    transform, crs = grid_profile["transform"], grid_profile["crs"]
    bounds = (transform.c, transform.f + transform.e * height, transform.c + transform.a * width, transform.f)
    with rasterio.open(source_path) as src:
        left, bottom, right, top = transform_bounds(crs, src.crs, *bounds, densify_pts=51)
        window = from_bounds(left - MARGIN, bottom - MARGIN, right + MARGIN, top + MARGIN, src.transform)
        window = window.round_offsets().round_lengths().intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=window).astype("float64")
        nodata = src.nodata
        source_transform = src.window_transform(window)
        source_crs = src.crs
    if nodata is not None:
        data[data == nodata] = 0.0
    data[data < 0] = 0.0
    out = np.zeros((height, width), dtype="float64")
    reproject(
        data, out, src_transform=source_transform, src_crs=source_crs, dst_transform=transform, dst_crs=crs,
        resampling=Resampling.sum, src_nodata=None, dst_nodata=None,
    )  # fmt: skip
    LOGGER.info("Source window %.4g persons, on the grid %.4g (%.3f).", data.sum(), out.sum(), out.sum() / data.sum())
    return out


def country_totals(population: np.ndarray, ids: np.ndarray, countries: list[str]) -> dict:
    """Population per country (cell-centre membership)."""
    valid = ids >= 0
    sums = np.bincount(ids[valid], weights=population[valid], minlength=len(countries))
    return dict(zip(countries, sums))


def main(zip_path, grid_path, polygons_path, output_path, reference=None):
    """Reproject one epoch to the grid, log the totals per country, write GeoTIFF."""
    ids, profile = read_raster(grid_path)
    countries = gpd.read_parquet(polygons_path)["country_id"].tolist()
    population = to_grid(tif_in_zip(zip_path), profile)
    for country, total in country_totals(population, ids, countries).items():
        ratio = total / reference[country] if reference and reference.get(country) else np.nan
        LOGGER.info("%s: %.4g persons (raster / Eurostat %.3f).", country, total, ratio)
    write_raster(output_path, population.astype("float32"), profile["transform"], profile["crs"], nodata=None)


if __name__ == "__main__":
    logging.basicConfig(filename=snakemake.log[0], level=logging.INFO, format="%(levelname)s %(message)s")  # noqa: F821
    main(
        zip_path=snakemake.input.zip,  # noqa: F821
        grid_path=snakemake.input.grid,  # noqa: F821
        polygons_path=snakemake.input.polygons,  # noqa: F821
        output_path=snakemake.output.population,  # noqa: F821
    )
