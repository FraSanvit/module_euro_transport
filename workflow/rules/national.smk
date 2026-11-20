"""Rules to combine national road vehicle-km and vehicle stock from all sources."""


rule combine_national_road_vkm:
    input:
        vkm=national_source_tables,
    output:
        vkm="<resources>/automatic/national_road_vkm.parquet",
    log:
        "<logs>/combine_national_road_vkm.log",
    conda:
        "../envs/module.yaml"
    params:
        metric="vkm",
        countries=config["countries"],
    message:
        "Combine national road vehicle-km of all configured countries and sources."
    script:
        "../scripts/combine_national_tables.py"


rule combine_national_road_stock:
    input:
        stock=national_stock_tables,
    output:
        stock="<resources>/automatic/national_road_stock.parquet",
    log:
        "<logs>/combine_national_road_stock.log",
    conda:
        "../envs/module.yaml"
    params:
        metric="stock",
        countries=config["countries"],
        priority=[
            internal["resources"]["automatic"][source]["source"]
            for source in internal["resources"]["automatic"]["stock_priority"]
        ],
    message:
        "Combine the national road vehicle stock of all configured countries, keeping per value the highest-priority source."
    script:
        "../scripts/combine_national_tables.py"
