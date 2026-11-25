"""Rules to combine national road vehicle-km and vehicle stock from all sources."""


rule combine_national_road_vkm:
    input:
        tables=national_source_tables,
        heavy_duty=lambda wc: rules.fill_heavy_duty_levels.output.levels,
    output:
        vkm="<resources>/automatic/national_road_vkm.parquet",
    log:
        "<logs>/combine_national_road_vkm.log",
    conda:
        "../envs/module.yaml"
    params:
        metric="vkm",
        countries=config["countries"],
        processed=lambda wc: hd_countries(),
    message:
        "Combine national road vehicle-km of all processed countries (configured countries and their neighbours) and sources; heavy_duty from the heavy-duty level chain."
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
        "Combine the national road vehicle stock of all processed countries (configured countries and their neighbours) and sources."
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
        countries=config["countries"],
        neighbours={},
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
        countries=config["countries"],
        neighbours=internal["gap_filling"]["neighbours"],
    message:
        "Fill gaps in the national road vehicle stock (interpolation of inner gaps, missing categories from neighbour ratios to cars)."
    script:
        "../scripts/fill_national_tables.py"
