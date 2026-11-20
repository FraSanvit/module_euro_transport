"""Rules to import national road vehicle-km from JRC-IDEES 2021."""


rule prepare_jrc_idees_road_vkm:
    message:
        "Extract road vehicle-km per vehicle category from JRC-IDEES 2021 for {wildcards.country}."
    params:
        jrc_code=lambda wc: internal["resources"]["automatic"]["jrc_idees"][
            "country_codes"
        ][wc.country],
        source=internal["resources"]["automatic"]["jrc_idees"]["source"],
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
    script:
        "../scripts/prepare_jrc_idees_road_vkm.py"


rule combine_jrc_idees_road_vkm:
    message:
        "Combine JRC-IDEES 2021 road vehicle-km for all configured countries."
    input:
        vkm=expand(
            "<resources>/automatic/jrc_idees/road_vkm/{country}.parquet",
            country=config["countries"],
        ),
    output:
        vkm="<resources>/automatic/jrc_idees/road_vkm.parquet",
    log:
        "<logs>/combine_jrc_idees_road_vkm.log",
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/combine_national_vkm.py"
