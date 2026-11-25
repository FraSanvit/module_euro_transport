"""Raster helpers for the heavy-duty spatial allocation (MULTIYEAR_PLAN.md §6.2, §6.10).

All rasters share one grid: EPSG:3035, square cells, north-up. A cell belongs to a
polygon if its centre is inside it (as in gregor).
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

NODATA_ID = -1


def grid_from_bounds(bounds, resolution: float, buffer: float = 0.0):
    """Transform and (rows, cols) of a grid covering `bounds` + `buffer`, snapped to `resolution`."""
    xmin, ymin, xmax, ymax = bounds
    xmin = np.floor((xmin - buffer) / resolution) * resolution
    ymin = np.floor((ymin - buffer) / resolution) * resolution
    xmax = np.ceil((xmax + buffer) / resolution) * resolution
    ymax = np.ceil((ymax + buffer) / resolution) * resolution
    shape = (int(round((ymax - ymin) / resolution)), int(round((xmax - xmin) / resolution)))
    return from_origin(xmin, ymax, resolution, resolution), shape


def rasterize_ids(geometries, transform, shape) -> np.ndarray:
    """Index (0..n-1) of the geometry containing each cell centre; NODATA_ID elsewhere."""
    pairs = [(geometry, i) for i, geometry in enumerate(geometries) if geometry is not None and not geometry.is_empty]
    return rasterize(pairs, out_shape=shape, transform=transform, fill=NODATA_ID, dtype="int32", all_touched=False)


def write_raster(path, array: np.ndarray, transform, crs, nodata=None) -> None:
    """Write a single-band GeoTIFF (deflate, tiled)."""
    is_float = np.issubdtype(array.dtype, np.floating)
    profile = {
        "driver": "GTiff", "height": array.shape[0], "width": array.shape[1], "count": 1,
        "dtype": array.dtype.name, "crs": crs, "transform": transform, "nodata": nodata,
        "compress": "deflate", "predictor": 3 if is_float else 2, "tiled": True,
        "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER",
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def read_raster(path):
    """Array and profile of a single-band raster."""
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def cell_centres(transform, shape):
    """Coordinates of the cell centres: x of the columns, y of the rows."""
    cols = transform.c + transform.a * (np.arange(shape[1]) + 0.5)
    rows = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
    return cols, rows
