"""Rules to combine national road vehicle-km from all sources."""


rule combine_national_road_vkm:
    input:
        vkm=national_source_tables,
    output:
        vkm="<resources>/automatic/national_road_vkm.parquet",
    log:
        "<logs>/combine_national_road_vkm.log",
    conda:
        "../envs/module.yaml"
    message:
        "Combine national road vehicle-km of all configured countries and sources."
    script:
        "../scripts/combine_national_vkm.py"
