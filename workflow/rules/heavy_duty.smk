"""Heavy-duty vehicle-km: national levels before gap filling (MULTIYEAR_PLAN.md §6.3-§6.8)."""


rule download_heavy_duty_eurostat:
    output:
        csv=f"{HD_DIR}/eurostat/{{name}}.csv",
    log:
        "<logs>/download_heavy_duty_eurostat/{name}.log",
    wildcard_constraints:
        name="|".join(HD["eurostat"]["queries"]),
    conda:
        "../envs/module.yaml"
    params:
        url=lambda wc: HD["eurostat"]["url"].format(
            query=HD["eurostat"]["queries"][wc.name]
        ),
    message:
        "Download the Eurostat table {wildcards.name} (SDMX-CSV)."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.csv} "{params.url}" 2> {log}'


rule download_heavy_duty_eurostat_ia:
    output:
        csv=f"{HD_DIR}/eurostat/road_go_ia_rc_{{year}}.csv",
    log:
        "<logs>/download_heavy_duty_eurostat_ia/{year}.log",
    wildcard_constraints:
        year=r"\d{4}",
    conda:
        "../envs/module.yaml"
    params:
        url=lambda wc: HD["eurostat"]["url"].format(
            query=HD["eurostat"]["road_go_ia_rc"].format(year=wc.year)
        ),
    message:
        "Download Eurostat road_go_ia_rc (international tonnes by haulier) for {wildcards.year}."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.csv} "{params.url}" 2> {log}'


rule download_etisplus:
    output:
        csv=f"{HD_DIR}/etisplus/{{file}}.csv",
    log:
        "<logs>/download_etisplus/{file}.log",
    wildcard_constraints:
        file="|".join(HD["etisplus"]["files"]),
    conda:
        "../envs/module.yaml"
    params:
        url=lambda wc: HD["etisplus"]["url"].format(
            file_id=HD["etisplus"]["files"][wc.file]["file_id"]
        ),
    message:
        "Download the ETISplus-based truck flow file {wildcards.file} (Mendeley Data v1)."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.csv} "{params.url}" 2> {log}'


rule download_heavy_duty_other:
    output:
        countries=f"{HD_DIR}/gisco/countries.geojson",
        dft=f"{HD_DIR}/dft_region_traffic.csv",
        trafa=f"{HD_DIR}/trafa_t04010.json",
    log:
        "<logs>/download_heavy_duty_other.log",
    conda:
        "../envs/module.yaml"
    params:
        countries=HD["gisco"]["countries"],
        dft=HD["territorial_totals"]["dft"]["url"],
        trafa=trafa_url(),
    message:
        "Download GISCO country polygons, DfT road traffic by region and Trafikanalys heavy-lorry vkm."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.countries} "{params.countries}" 2> {log} && '
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.dft} "{params.dft}" 2>> {log} && '
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.trafa} "{params.trafa}" 2>> {log}'


rule download_heavy_duty_zenodo:
    output:
        record=f"{HD_DIR}/zenodo/national_data.zip",
    log:
        "<logs>/download_heavy_duty_zenodo.log",
    conda:
        "../envs/module.yaml"
    params:
        url=HD["zenodo"]["url"],
    message:
        "Download the project's Zenodo record of static inputs (German toll series)."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.record} "{params.url}" 2> {log}'


rule split_etisplus_od_flows:
    input:
        od_flows=f"{HD_DIR}/etisplus/od_flows.csv",
        regions=f"{HD_DIR}/etisplus/regions.csv",
        edges=f"{HD_DIR}/etisplus/edges.csv",
    output:
        edges=f"{HD_DIR}/etisplus_edges_split.parquet",
    log:
        "<logs>/split_etisplus_od_flows.log",
    conda:
        "../envs/module.yaml"
    params:
        sha256={k: v["sha256"] for k, v in HD["etisplus"]["files"].items()},
        flow_column=HD["etisplus"]["flow_column"],
        chunk_rows=HD["etisplus"]["chunk_rows"],
    message:
        "Split ETISplus edge flows into domestic and international OD flows (about 10 minutes)."
    script:
        "../scripts/split_etisplus_od_flows.py"


rule prepare_heavy_duty_network:
    input:
        edges=rules.split_etisplus_od_flows.output.edges,
        nodes=f"{HD_DIR}/etisplus/nodes.csv",
        countries=f"{HD_DIR}/gisco/countries.geojson",
    output:
        network=f"{HD_DIR}/network_by_country.parquet",
    log:
        "<logs>/prepare_heavy_duty_network.log",
    conda:
        "../envs/module.yaml"
    params:
        nodes_sha256=HD["etisplus"]["files"]["nodes"]["sha256"],
        codes=HD["country_codes"],
        crs=HD["crs"],
        ferry_sea_share=HD["etisplus"]["ferry_sea_share"],
        sea_handling=HD["etisplus"]["sea_handling"],
    message:
        "ETISplus network vkm per country (overlay with GISCO country polygons)."
    script:
        "../scripts/prepare_heavy_duty_network.py"


rule prepare_heavy_duty_statistics:
    input:
        tott=f"{HD_DIR}/eurostat/road_go_ta_tott.csv",
        dc=f"{HD_DIR}/eurostat/road_go_ta_dc.csv",
        hac=f"{HD_DIR}/eurostat/road_go_ca_hac.csv",
        tra=f"{HD_DIR}/eurostat/road_go_ta_tra.csv",
        vm=f"{HD_DIR}/eurostat/road_go_ta_vm.csv",
        ia=lambda wc: [f"{HD_DIR}/eurostat/road_go_ia_rc_{y}.csv" for y in hd_years()],
    output:
        statistics=f"{HD_DIR}/statistics.parquet",
    log:
        "<logs>/prepare_heavy_duty_statistics.log",
    conda:
        "../envs/module.yaml"
    params:
        codes=lambda wc: {c: HD["country_codes"][c] for c in hd_countries()},
        years=hd_years(),
        reference_year=config["heavy_duty"]["reference_year"],
        include_cabotage=config["heavy_duty"]["cabotage"]["include"],
        empty_share=config["heavy_duty"]["cabotage"]["empty_share"],
        cabotage_zero=HD["cabotage_zero"],
        exclude=HD["exclude"],
        max_coverage_gap=HD["max_coverage_gap"],
        empty_bandwidth=HD["empty_km_bandwidth"],
        correct_coverage=config["heavy_duty"]["international"][
            "reporter_coverage_correction"
        ],
    message:
        "Heavy-duty statistics per country and year from Eurostat road freight tables."
    script:
        "../scripts/prepare_heavy_duty_statistics.py"


rule prepare_heavy_duty_territorial_totals:
    input:
        unpack(hd_totals_inputs),
    output:
        totals=f"{HD_DIR}/territorial_totals.parquet",
    log:
        "<logs>/prepare_heavy_duty_territorial_totals.log",
    conda:
        "../envs/module.yaml"
    params:
        settings=HD["territorial_totals"],
        countries=lambda wc: [
            c for c in hd_countries() if c in HD["territorial_totals"]["overrides"]
        ],
        years=hd_years(),
    message:
        "Territorial national totals of heavy-duty vkm (GBR, SWE, CHE)."
    script:
        "../scripts/prepare_heavy_duty_territorial_totals.py"


rule prepare_heavy_duty_toll:
    input:
        record=rules.download_heavy_duty_zenodo.output.record,
    output:
        toll=f"{HD_DIR}/toll.parquet",
    log:
        "<logs>/prepare_heavy_duty_toll.log",
    conda:
        "../envs/module.yaml"
    params:
        md5=HD["zenodo"]["md5"],
        member=HD["zenodo"]["toll_file"],
        sha256=HD["zenodo"]["toll_sha256"],
    message:
        "German motorway toll km by vehicle nationality (Zenodo record)."
    script:
        "../scripts/prepare_heavy_duty_toll.py"


rule heavy_duty_national_levels:
    input:
        statistics=rules.prepare_heavy_duty_statistics.output.statistics,
        network=rules.prepare_heavy_duty_network.output.network,
        totals=rules.prepare_heavy_duty_territorial_totals.output.totals,
        toll=lambda wc: (
            [rules.prepare_heavy_duty_toll.output.toll]
            if set(HD["toll_countries"]) & set(hd_countries())
            else []
        ),
    output:
        levels=f"{HD_DIR}/national_levels.parquet",
        calibration=f"{HD_DIR}/calibration.parquet",
    log:
        "<logs>/heavy_duty_national_levels.log",
    conda:
        "../envs/module.yaml"
    params:
        codes=HD["country_codes"],
        countries=lambda wc: hd_countries(),
        years=hd_years(),
        reference_year=config["heavy_duty"]["reference_year"],
        parameters=config["heavy_duty"]["parameters"],
        islands=HD["islands"],
        toll_countries=HD["toll_countries"],
        toll_cabotage_share=HD["toll_cabotage_share"],
        jump=HD["jump_flag"],
    message:
        "Heavy-duty national levels per component and year, before gap filling."
    script:
        "../scripts/heavy_duty_national_levels.py"


rule heavy_duty_manifest:
    input:
        files=lambda wc: list(hd_downloads()),
    output:
        manifest=f"{HD_DIR}/manifest.csv",
    log:
        "<logs>/heavy_duty_manifest.log",
    conda:
        "../envs/module.yaml"
    params:
        urls=lambda wc: list(hd_downloads().values()),
    message:
        "Checksum manifest of the heavy-duty downloads (path, URL, size, SHA-256)."
    script:
        "../scripts/heavy_duty_manifest.py"
