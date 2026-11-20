"""Rules to used to download automatic resource files."""


rule download_jrc_idees:
    message:
        "Download the JRC-IDEES 2021 country zip for {wildcards.country}."
    params:
        url=lambda wc: internal["resources"]["automatic"]["jrc_idees"]["url"].format(
            jrc_code=internal["resources"]["automatic"]["jrc_idees"]["country_codes"][
                wc.country
            ]
        ),
    output:
        zip="<resources>/automatic/jrc_idees/raw/{country}.zip",
    log:
        "<logs>/download_jrc_idees/{country}.log",
    wildcard_constraints:
        country="[A-Z]{3}",
    conda:
        "../envs/module.yaml"
    shell:
        'curl -sSfL --retry 3 -o {output.zip} "{params.url}" 2> {log}'
