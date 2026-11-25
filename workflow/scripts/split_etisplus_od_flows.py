"""Split ETISplus edge flows into domestic and international OD flows (MULTIYEAR_PLAN.md §6.1 step 1).

Each OD row is domestic if its origin and destination zones are in the same country
(02_NUTS-3-Regions `Country`), else international (incl. transit). Its trucks are added
to every edge of its path. The first and last legs (to and from the E-road network,
not on any edge) are summed per zone in the same pass (§6.1 step 4). Inputs are checked
against the SHA-256 published by Mendeley.
Runtime about 10 minutes (2.8 GB of OD rows, read in chunks).
"""

import logging

import pandas as pd
from _etisplus import (
    LEG_COLUMNS,
    check_split,
    legs_by_zone,
    read_edges,
    split_flows,
    zone_legs,
)
from _heavy_duty import verify_sha256

LOGGER = logging.getLogger(__name__)
CHECK_TOLERANCE = 0.01


def main(od_path, regions_path, edges_path, sha256, flow_column, chunk_rows, output_path, legs_path):
    """Verify inputs, split flows per edge and write parquet."""
    for key, path in {"od_flows": od_path, "regions": regions_path, "edges": edges_path}.items():
        verify_sha256(path, sha256[key])
    regions = pd.read_csv(regions_path)
    zone_country = regions.set_index("ETISPlus_Zone_ID")["Country"]
    edges = read_edges(edges_path)
    columns = ["ID_origin_region", "ID_destination_region", "Edge_path_E_road", flow_column, *LEG_COLUMNS]
    legs = []

    def with_legs(chunks):
        for chunk in chunks:
            legs.append(zone_legs(chunk, zone_country, flow_column))
            yield chunk

    chunks = pd.read_csv(od_path, chunksize=chunk_rows, usecols=columns)
    split = split_flows(with_legs(chunks), zone_country, edges, flow_column)
    ratio = check_split(split, flow_column)
    if abs(ratio - 1) > CHECK_TOLERANCE:
        raise ValueError(f"Rebuilt edge flows differ from published flows by {ratio - 1:+.2%}.")
    split.to_parquet(output_path, index=False)
    legs_by_zone(legs, regions).to_parquet(legs_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        od_path=snakemake.input.od_flows,  # noqa: F821
        regions_path=snakemake.input.regions,  # noqa: F821
        edges_path=snakemake.input.edges,  # noqa: F821
        sha256=snakemake.params.sha256,  # noqa: F821
        flow_column=snakemake.params.flow_column,  # noqa: F821
        chunk_rows=snakemake.params.chunk_rows,  # noqa: F821
        output_path=snakemake.output.edges,  # noqa: F821
        legs_path=snakemake.output.legs,  # noqa: F821
    )
