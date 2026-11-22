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
    message:
        "Combine the national road vehicle stock of all configured countries and sources."
    script:
        "../scripts/combine_national_tables.py"


rule fill_national_road_vkm:
    input:
        vkm=rules.combine_national_road_vkm.output.vkm,
    output:
        vkm="<resources>/automatic/national_road_vkm_filled.parquet",
    log:
        "<logs>/fill_national_road_vkm.log",
    conda:
        "../envs/module.yaml"
    params:
        metric="vkm",
        max_years=internal["gap_filling"]["interpolation_max_years"],
    message:
        "Fill gaps in the national road vehicle-km (interpolation of inner gaps)."
    script:
        "../scripts/fill_national_tables.py"


rule fill_national_road_stock:
    input:
        stock=rules.combine_national_road_stock.output.stock,
    output:
        stock="<resources>/automatic/national_road_stock_filled.parquet",
    log:
        "<logs>/fill_national_road_stock.log",
    conda:
        "../envs/module.yaml"
    params:
        metric="stock",
        max_years=internal["gap_filling"]["interpolation_max_years"],
    message:
        "Fill gaps in the national road vehicle stock (interpolation of inner gaps)."
    script:
        "../scripts/fill_national_tables.py"
