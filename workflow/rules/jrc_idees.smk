"""Rules to import national road vehicle-km and vehicle stock from JRC-IDEES 2021."""


rule prepare_jrc_idees_road_vkm:
    input:
        zip=rules.download_jrc_idees.output.zip,
    output:
        vkm="<resources>/automatic/jrc_idees/road_vkm/{country}.parquet",
    log:
        "<logs>/prepare_jrc_idees_road_vkm/{country}.log",
    wildcard_constraints:
        country="[A-Z]{3}",
    conda:
        "../envs/module.yaml"
    params:
        jrc_code=lambda wc: internal["resources"]["automatic"]["jrc_idees"][
            "country_codes"
        ][wc.country],
        source=internal["resources"]["automatic"]["jrc_idees"]["source"],
    message:
        "Extract road vehicle-km per vehicle category from JRC-IDEES 2021 for {wildcards.country}."
    script:
        "../scripts/prepare_jrc_idees_road_vkm.py"


rule combine_jrc_idees_road_vkm:
    input:
        vkm=expand(
            "<resources>/automatic/jrc_idees/road_vkm/{country}.parquet",
            country=countries_from("jrc_idees"),
        ),
    output:
        vkm="<resources>/automatic/jrc_idees/road_vkm.parquet",
    log:
        "<logs>/combine_jrc_idees_road_vkm.log",
    conda:
        "../envs/module.yaml"
    params:
        metric="vkm",
        countries=countries_from("jrc_idees"),
    message:
        "Combine JRC-IDEES 2021 road vehicle-km for all configured countries."
    script:
        "../scripts/combine_national_tables.py"


rule prepare_jrc_idees_road_stock:
    input:
        zip=rules.download_jrc_idees.output.zip,
    output:
        stock="<resources>/automatic/jrc_idees/road_stock/{country}.parquet",
    log:
        "<logs>/prepare_jrc_idees_road_stock/{country}.log",
    wildcard_constraints:
        country="[A-Z]{3}",
    conda:
        "../envs/module.yaml"
    params:
        jrc_code=lambda wc: internal["resources"]["automatic"]["jrc_idees"][
            "country_codes"
        ][wc.country],
        source=internal["resources"]["automatic"]["jrc_idees"]["source"],
    message:
        "Extract the road vehicle stock per vehicle category from JRC-IDEES 2021 for {wildcards.country}."
    script:
        "../scripts/prepare_jrc_idees_road_stock.py"


rule combine_jrc_idees_road_stock:
    input:
        stock=expand(
            "<resources>/automatic/jrc_idees/road_stock/{country}.parquet",
            country=stock_countries_from("jrc_idees"),
        ),
    output:
        stock="<resources>/automatic/jrc_idees/road_stock.parquet",
    log:
        "<logs>/combine_jrc_idees_road_stock.log",
    conda:
        "../envs/module.yaml"
    params:
        metric="stock",
        countries=stock_countries_from("jrc_idees"),
    message:
        "Combine the JRC-IDEES 2021 road vehicle stock for all configured countries."
    script:
        "../scripts/combine_national_tables.py"
