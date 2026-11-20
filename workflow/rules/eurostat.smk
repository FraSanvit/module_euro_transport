"""Rules to import national road vehicle-km (non-EU countries) and vehicle stock from Eurostat."""


rule download_eurostat:
    output:
        csv="<resources>/automatic/eurostat/raw/{dataset}.csv",
    log:
        "<logs>/download_eurostat/{dataset}.log",
    wildcard_constraints:
        dataset="road_tf_vehmov|road_tf_road|road_eqs_lorroa|road_eqs_carmot|road_eqs_busveh|road_eqs_mopeds",
    conda:
        "../envs/module.yaml"
    params:
        url=lambda wc: internal["resources"]["automatic"]["eurostat"]["url"].format(
            dataset=wc.dataset
        ),
    message:
        "Download the Eurostat dataset {wildcards.dataset} as SDMX-CSV."
    shell:
        'curl -sSfL --retry 3 -o {output.csv} "{params.url}" 2> {log}'


rule prepare_eurostat_road_vkm:
    input:
        road_tf_vehmov="<resources>/automatic/eurostat/raw/road_tf_vehmov.csv",
        road_tf_road="<resources>/automatic/eurostat/raw/road_tf_road.csv",
    output:
        vkm="<resources>/automatic/eurostat/road_vkm.parquet",
    log:
        "<logs>/prepare_eurostat_road_vkm.log",
    conda:
        "../envs/module.yaml"
    params:
        country_codes=lambda wc: {
            country: internal["resources"]["automatic"]["eurostat"]["country_codes"][
                country
            ]
            for country in countries_from("eurostat")
        },
        principles=internal["resources"]["automatic"]["eurostat"]["principle"],
        categories=internal["resources"]["automatic"]["eurostat"]["vkm_categories"],
        source=internal["resources"]["automatic"]["eurostat"]["source"],
    message:
        "Derive road vehicle-km per vehicle category from Eurostat for the configured non-EU countries."
    script:
        "../scripts/prepare_eurostat_road_vkm.py"


rule prepare_eurostat_road_stock:
    input:
        road_eqs_lorroa="<resources>/automatic/eurostat/raw/road_eqs_lorroa.csv",
        road_eqs_carmot="<resources>/automatic/eurostat/raw/road_eqs_carmot.csv",
        road_eqs_busveh="<resources>/automatic/eurostat/raw/road_eqs_busveh.csv",
        road_eqs_mopeds="<resources>/automatic/eurostat/raw/road_eqs_mopeds.csv",
    output:
        stock="<resources>/automatic/eurostat/road_stock.parquet",
    log:
        "<logs>/prepare_eurostat_road_stock.log",
    conda:
        "../envs/module.yaml"
    params:
        geo_codes=lambda wc: {
            country: internal["resources"]["automatic"]["eurostat_stock"][
                "stock_country_codes"
            ][country]
            for country in stock_countries_from("eurostat_stock")
        },
        source=internal["resources"]["automatic"]["eurostat_stock"]["source"],
        exclude=internal["resources"]["automatic"]["eurostat_stock"].get("exclude", {}),
    message:
        "Derive the road vehicle stock per vehicle category from Eurostat for the countries routed to it."
    script:
        "../scripts/prepare_eurostat_road_stock.py"
