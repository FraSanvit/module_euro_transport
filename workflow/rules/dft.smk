"""Rules to import the national road vehicle stock for Great Britain from DfT."""


rule download_dft:
    output:
        ods="<resources>/automatic/dft/raw/veh0101.ods",
    log:
        "<logs>/download_dft.log",
    conda:
        "../envs/module.yaml"
    params:
        url=internal["resources"]["automatic"]["dft"]["url"],
    message:
        "Download the DfT vehicle licensing table VEH0101."
    shell:
        'curl -sSfL --retry 3 -o {output.ods} "{params.url}" 2> {log}'


rule prepare_dft_road_stock:
    input:
        ods="<resources>/automatic/dft/raw/veh0101.ods",
    output:
        stock="<resources>/automatic/dft/road_stock.parquet",
    log:
        "<logs>/prepare_dft_road_stock.log",
    conda:
        "../envs/module.yaml"
    params:
        country_codes=internal["resources"]["automatic"]["dft"]["stock_country_codes"],
        source=internal["resources"]["automatic"]["dft"]["source"],
    message:
        "Derive the road vehicle stock per vehicle category for Great Britain from DfT."
    script:
        "../scripts/prepare_dft_road_stock.py"
