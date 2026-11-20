"""Rules to import national road vehicle-km and vehicle stock for Switzerland from BFS."""


rule download_bfs:
    output:
        xlsx="<resources>/automatic/bfs/raw/{table}.xlsx",
    log:
        "<logs>/download_bfs/{table}.log",
    wildcard_constraints:
        table="freight|passenger|stock|mopeds",
    conda:
        "../envs/module.yaml"
    params:
        url=lambda wc: internal["resources"]["automatic"]["bfs"]["url"].format(
            asset_id=internal["resources"]["automatic"]["bfs"]["tables"][wc.table][
                "asset_id"
            ]
        ),
    message:
        "Download the BFS {wildcards.table} table."
    shell:
        'curl -sSfL --retry 3 -o {output.xlsx} "{params.url}" 2> {log}'


rule prepare_bfs_road_vkm:
    input:
        freight="<resources>/automatic/bfs/raw/freight.xlsx",
        passenger="<resources>/automatic/bfs/raw/passenger.xlsx",
    output:
        vkm="<resources>/automatic/bfs/road_vkm.parquet",
    log:
        "<logs>/prepare_bfs_road_vkm.log",
    conda:
        "../envs/module.yaml"
    params:
        country_codes=internal["resources"]["automatic"]["bfs"]["country_codes"],
        principles=internal["resources"]["automatic"]["bfs"]["principle"],
        source=internal["resources"]["automatic"]["bfs"]["source"],
    message:
        "Derive road vehicle-km per vehicle category for Switzerland from BFS."
    script:
        "../scripts/prepare_bfs_road_vkm.py"


rule prepare_bfs_road_stock:
    input:
        stock="<resources>/automatic/bfs/raw/stock.xlsx",
        mopeds="<resources>/automatic/bfs/raw/mopeds.xlsx",
    output:
        stock="<resources>/automatic/bfs/road_stock.parquet",
    log:
        "<logs>/prepare_bfs_road_stock.log",
    conda:
        "../envs/module.yaml"
    params:
        country_codes=internal["resources"]["automatic"]["bfs"]["country_codes"],
        source=internal["resources"]["automatic"]["bfs"]["stock_source"],
    message:
        "Derive the road vehicle stock per vehicle category for Switzerland from BFS."
    script:
        "../scripts/prepare_bfs_road_stock.py"
