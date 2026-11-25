"""NUTS-3 regions on the allocation grid, one raster per NUTS version (MULTIYEAR_PLAN.md §6.10).

Each cell gets the index of the NUTS-3 region containing its centre (-1 outside); the
codes are written alongside. Also writes the NUTS-3 tonnes table (loaded + unloaded) / 2.
"""

import logging

import geopandas as gpd
import pandas as pd
from _heavy_duty import read_eurostat
from _raster import rasterize_ids, read_raster, write_raster
from _short_haul import tonnes_table

LOGGER = logging.getLogger(__name__)


def main(nuts_paths: dict, grid_path, loaded_path, unloaded_path, raster_paths: dict, codes_path, tonnes_path):
    """Rasterise each NUTS version and prepare the tonnes table."""
    _, profile = read_raster(grid_path)
    shape, transform = (profile["height"], profile["width"]), profile["transform"]
    rows = []
    for version, path in nuts_paths.items():
        regions = gpd.read_file(path).to_crs(profile["crs"]).sort_values("NUTS_ID", ignore_index=True)
        ids = rasterize_ids(regions.geometry, transform, shape)
        write_raster(raster_paths[version], ids, transform, profile["crs"], nodata=-1)
        rows.append(pd.DataFrame({"version": str(version), "index": range(len(regions)), "nuts": regions["NUTS_ID"]}))
        LOGGER.info("NUTS %s: %d regions, %d cells.", version, len(regions), int((ids >= 0).sum()))
    pd.concat(rows, ignore_index=True).to_parquet(codes_path, index=False)
    loaded = read_eurostat(loaded_path, ["geo"], {"nst07": "TOTAL", "unit": "THS_T"})
    unloaded = read_eurostat(unloaded_path, ["geo"], {"nst07": "TOTAL", "unit": "THS_T"})
    tonnes = tonnes_table(loaded, unloaded)
    LOGGER.info("NUTS-3 tonnes: %d regions, years %d-%d.", tonnes["nuts"].nunique(), tonnes["year"].min(), tonnes["year"].max())
    tonnes.to_parquet(tonnes_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(filename=snakemake.log[0], level=logging.INFO, format="%(levelname)s %(message)s")  # noqa: F821
    versions = snakemake.params.versions  # noqa: F821
    main(
        nuts_paths={v: snakemake.input[f"nuts_{v}"] for v in versions},  # noqa: F821
        grid_path=snakemake.input.grid,  # noqa: F821
        loaded_path=snakemake.input.loaded,  # noqa: F821
        unloaded_path=snakemake.input.unloaded,  # noqa: F821
        raster_paths={v: snakemake.output[f"raster_{v}"] for v in versions},  # noqa: F821
        codes_path=snakemake.output.codes,  # noqa: F821
        tonnes_path=snakemake.output.tonnes,  # noqa: F821
    )
