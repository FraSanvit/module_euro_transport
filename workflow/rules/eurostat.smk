"""Rules to import national road vehicle-km from Eurostat (non-EU countries)."""


rule download_eurostat:
    output:
        csv="<resources>/automatic/eurostat/raw/{dataset}.csv",
    log:
        "<logs>/download_eurostat/{dataset}.log",
    wildcard_constraints:
        dataset="road_tf_vehmov|road_tf_road",
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
        source=internal["resources"]["automatic"]["eurostat"]["source"],
    message:
        "Derive road vehicle-km per vehicle category from Eurostat for the configured non-EU countries."
    script:
        "../scripts/prepare_eurostat_road_vkm.py"
