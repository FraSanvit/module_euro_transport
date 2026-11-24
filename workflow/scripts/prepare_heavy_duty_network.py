"""ETISplus network vkm per country, domestic and international OD flows (MULTIYEAR_PLAN.md §6.1 step 3).

Edges (straight lines between nodes) are overlaid with GISCO country polygons in an
equal-area projection; sea parts go half to each end node's country. Reference flows
(ETISplus 2019); used only for the reference-year calibration (§6.3).
"""

import logging

import geopandas as gpd
import pandas as pd
from _etisplus import network_vkm_by_country
from _heavy_duty import verify_sha256

LOGGER = logging.getLogger(__name__)


def to_iso3(network: pd.DataFrame, codes: dict) -> pd.DataFrame:
    """Keep the module's countries and add `country_id` (ISO3); log countries without network."""
    to_iso = {geo: iso for iso, geo in codes.items()}
    kept = network[network["geo"].isin(to_iso)].copy()
    kept.insert(0, "country_id", kept["geo"].map(to_iso))
    missing = sorted(set(codes) - set(kept["country_id"]))
    if missing:
        LOGGER.warning("No ETISplus network vkm for %s (no polygon or no network).", missing)
    return kept.sort_values("country_id", ignore_index=True)


def main(edges_path, nodes_path, nodes_sha256, countries_path, codes, crs, ferry_sea_share, output_path):
    """Overlay the network with country polygons and write parquet."""
    verify_sha256(nodes_path, nodes_sha256)
    edges = pd.read_parquet(edges_path)
    nodes = pd.read_csv(nodes_path)
    countries = gpd.read_file(countries_path)
    network = network_vkm_by_country(edges, nodes, countries, crs, ferry_sea_share)
    to_iso3(network, codes).to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        edges_path=snakemake.input.edges,  # noqa: F821
        nodes_path=snakemake.input.nodes,  # noqa: F821
        nodes_sha256=snakemake.params.nodes_sha256,  # noqa: F821
        countries_path=snakemake.input.countries,  # noqa: F821
        codes=snakemake.params.codes,  # noqa: F821
        crs=snakemake.params.crs,  # noqa: F821
        ferry_sea_share=snakemake.params.ferry_sea_share,  # noqa: F821
        output_path=snakemake.output.network,  # noqa: F821
    )
