"""Rules to import national road data from statistics offices with a PX-Web API.

MAKSTAT (North Macedonia), Statistics Iceland (Hagstofa) and ASK (Kosovo).
"""


rule download_pxweb:
    output:
        csv="<resources>/automatic/{provider}/raw/{table}.csv",
        listing="<resources>/automatic/{provider}/raw/{table}_listing.json",
    log:
        "<logs>/download_pxweb/{provider}/{table}.log",
    wildcard_constraints:
        provider="makstat|hagstofa|ask",
        table="freight|stock",
    conda:
        "../envs/module.yaml"
    params:
        url=lambda wc: pxweb_url(wc.provider, wc.table),
        folder_url=lambda wc: pxweb_url(wc.provider, wc.table).rsplit("/", 1)[0],
        # Request body: the full table as CSV (no filter).
        query=workflow.source_path("../internal/pxweb_query.json"),
    message:
        "Download the {wildcards.provider} {wildcards.table} table as CSV and its folder listing."
    shell:
        'curl -sSf --retry 3 -X POST -H "Content-Type: application/json" -d @{params.query} -o {output.csv} "{params.url}" 2> {log} && '
        'curl -sSf --retry 3 -o {output.listing} "{params.folder_url}" 2>> {log}'


rule prepare_makstat_road_vkm:
    input:
        freight="<resources>/automatic/makstat/raw/freight.csv",
        listing="<resources>/automatic/makstat/raw/freight_listing.json",
    output:
        vkm="<resources>/automatic/makstat/road_vkm.parquet",
    log:
        "<logs>/prepare_makstat_road_vkm.log",
    conda:
        "../envs/module.yaml"
    params:
        table_id=internal["resources"]["automatic"]["makstat"]["tables"]["freight"][
            "table_id"
        ],
        country_codes=internal["resources"]["automatic"]["makstat"]["country_codes"],
        principles=internal["resources"]["automatic"]["makstat"]["principle"],
        source=internal["resources"]["automatic"]["makstat"]["source"],
    message:
        "Derive heavy-duty road vehicle-km for North Macedonia from MAKSTAT."
    script:
        "../scripts/prepare_makstat_road_vkm.py"


rule prepare_makstat_road_stock:
    input:
        stock="<resources>/automatic/makstat/raw/stock.csv",
        listing="<resources>/automatic/makstat/raw/stock_listing.json",
    output:
        stock="<resources>/automatic/makstat/road_stock.parquet",
    log:
        "<logs>/prepare_makstat_road_stock.log",
    conda:
        "../envs/module.yaml"
    params:
        table_id=internal["resources"]["automatic"]["makstat"]["tables"]["stock"][
            "table_id"
        ],
        country_codes=internal["resources"]["automatic"]["makstat"]["country_codes"],
        source=internal["resources"]["automatic"]["makstat"]["stock_source"],
    message:
        "Derive the road vehicle stock per vehicle category for North Macedonia from MAKSTAT."
    script:
        "../scripts/prepare_makstat_road_stock.py"


rule prepare_hagstofa_road_stock:
    input:
        stock="<resources>/automatic/hagstofa/raw/stock.csv",
        listing="<resources>/automatic/hagstofa/raw/stock_listing.json",
    output:
        stock="<resources>/automatic/hagstofa/road_stock.parquet",
    log:
        "<logs>/prepare_hagstofa_road_stock.log",
    conda:
        "../envs/module.yaml"
    params:
        table_id=internal["resources"]["automatic"]["hagstofa"]["tables"]["stock"][
            "table_id"
        ],
        country_codes=internal["resources"]["automatic"]["hagstofa"][
            "stock_country_codes"
        ],
        source=internal["resources"]["automatic"]["hagstofa"]["source"],
    message:
        "Derive the road vehicle stock per vehicle category for Iceland from Statistics Iceland."
    script:
        "../scripts/prepare_hagstofa_road_stock.py"


rule prepare_ask_road_stock:
    input:
        stock="<resources>/automatic/ask/raw/stock.csv",
        listing="<resources>/automatic/ask/raw/stock_listing.json",
    output:
        stock="<resources>/automatic/ask/road_stock.parquet",
    log:
        "<logs>/prepare_ask_road_stock.log",
    conda:
        "../envs/module.yaml"
    params:
        table_id=internal["resources"]["automatic"]["ask"]["tables"]["stock"][
            "table_id"
        ],
        country_codes=internal["resources"]["automatic"]["ask"]["stock_country_codes"],
        source=internal["resources"]["automatic"]["ask"]["source"],
        exclude=internal["resources"]["automatic"]["ask"]["exclude"],
    message:
        "Derive the road vehicle stock per vehicle category for Kosovo from ASK."
    script:
        "../scripts/prepare_ask_road_stock.py"
