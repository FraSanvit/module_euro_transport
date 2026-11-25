"""Population raster on the allocation grid for one GHS-POP epoch (MULTIYEAR_PLAN.md §6.2).

GHS-POP R2023A, 1 km, Mollweide (ESRI:54009). The part covering the grid is read from
the zip, nodata set to 0, and summed into the grid cells by sub-cell points (persons per cell,
mass-conserving).
Years between epochs are interpolated later (`population_for_year`).
"""

import logging
import zipfile

import geopandas as gpd
import numpy as np
import rasterio
from _raster import read_raster, write_raster
from pyproj import Transformer
from rasterio.warp import transform_bounds
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


def to_grid(source_path, grid_profile, split: int = 3, chunk_rows: int = 256) -> np.ndarray:
    """Population of the source raster summed into the grid cells (float64), mass-conserving.

    Each populated source cell is split into `split` x `split` sub-points carrying equal
    shares of its population; each sub-point is added to the grid cell containing it.
    (GDAL's `sum` resampling dropped whole blocks: 1.2 M persons in Austria.)
    """
    height, width = grid_profile["height"], grid_profile["width"]
    transform, crs = grid_profile["transform"], grid_profile["crs"]
    bounds = (transform.c, transform.f + transform.e * height, transform.c + transform.a * width, transform.f)
    out = np.zeros(height * width, dtype="float64")
    with rasterio.open(source_path) as src:
        left, bottom, right, top = transform_bounds(crs, src.crs, *bounds, densify_pts=51)
        window = from_bounds(left - MARGIN, bottom - MARGIN, right + MARGIN, top + MARGIN, src.transform)
        window = window.round_offsets().round_lengths().intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=window).astype("float64")
        nodata, st = src.nodata, src.window_transform(window)
        to_grid_crs = Transformer.from_crs(src.crs, crs, always_xy=True)
    if nodata is not None:
        data[data == nodata] = 0.0
    data[~np.isfinite(data) | (data < 0)] = 0.0
    offsets = (np.arange(split) + 0.5) / split
    dx, dy = np.meshgrid(offsets, offsets)
    dx, dy = dx.ravel(), dy.ravel()
    placed = 0.0
    for r0 in range(0, data.shape[0], chunk_rows):
        rows, cols = np.nonzero(data[r0 : r0 + chunk_rows] > 0)
        if rows.size == 0:
            continue
        values = data[r0 + rows, cols]
        sx = st.c + st.a * (cols[:, None] + dx[None, :])
        sy = st.f + st.e * (r0 + rows[:, None] + dy[None, :])
        gx, gy = to_grid_crs.transform(sx.ravel(), sy.ravel())
        col = np.floor((np.asarray(gx) - transform.c) / transform.a).astype(np.int64)
        row = np.floor((np.asarray(gy) - transform.f) / transform.e).astype(np.int64)
        share = np.repeat(values / split**2, split**2)
        inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        np.add.at(out, row[inside] * width + col[inside], share[inside])
        placed += share[inside].sum()
    LOGGER.info("Source window %.4g persons, on the grid %.4g (%.3f).", data.sum(), placed, placed / data.sum())
    return out.reshape(height, width)


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
