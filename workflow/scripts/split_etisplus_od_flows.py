"""Split ETISplus edge flows into domestic and international OD flows (MULTIYEAR_PLAN.md §6.1 step 1).

Each OD row is domestic if its origin and destination zones are in the same country
(02_NUTS-3-Regions `Country`), else international (incl. transit). Its trucks are added
to every edge of its path. Inputs are checked against the SHA-256 published by Mendeley.
Runtime about 10 minutes (2.8 GB of OD rows, read in chunks).
"""

import logging

import pandas as pd
from _etisplus import check_split, read_edges, split_flows
from _heavy_duty import verify_sha256

LOGGER = logging.getLogger(__name__)
CHECK_TOLERANCE = 0.01


def main(od_path, regions_path, edges_path, sha256, flow_column, chunk_rows, output_path):
    """Verify inputs, split flows per edge and write parquet."""
    for key, path in {"od_flows": od_path, "regions": regions_path, "edges": edges_path}.items():
        verify_sha256(path, sha256[key])
    zone_country = pd.read_csv(regions_path).set_index("ETISPlus_Zone_ID")["Country"]
    edges = read_edges(edges_path)
    columns = ["ID_origin_region", "ID_destination_region", "Edge_path_E_road", flow_column]
    chunks = pd.read_csv(od_path, chunksize=chunk_rows, usecols=columns)
    split = split_flows(chunks, zone_country, edges, flow_column)
    ratio = check_split(split, flow_column)
    if abs(ratio - 1) > CHECK_TOLERANCE:
        raise ValueError(f"Rebuilt edge flows differ from published flows by {ratio - 1:+.2%}.")
    split.to_parquet(output_path, index=False)


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
    )
