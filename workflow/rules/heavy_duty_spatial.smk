"""Heavy-duty vehicle-km: spatial allocation (MULTIYEAR_PLAN.md §6.1-§6.2, §6.10)."""

SPATIAL_DIR = f"{HD_DIR}/spatial"


rule download_heavy_duty_spatial_countries:
    output:
        countries=f"{SPATIAL_DIR}/raw/countries_03m.geojson",
    log:
        "<logs>/download_heavy_duty_spatial_countries.log",
    conda:
        "../envs/module.yaml"
    params:
        url=HD["spatial"]["countries"],
    message:
        "Download GISCO country polygons (1:3 million) for the allocation grid."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.countries} "{params.url}" 2> {log}'


rule download_heavy_duty_spatial_carved:
    output:
        shape=f"{SPATIAL_DIR}/raw/carved_{{country}}.geojson",
    log:
        "<logs>/download_heavy_duty_spatial_carved_{country}.log",
    conda:
        "../envs/module.yaml"
    params:
        url=lambda wc: HD["spatial"]["carved"][wc.country]["url"],
    message:
        "Download the polygon of {wildcards.country}, which GISCO does not separate."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.shape} "{params.url}" 2> {log}'


def carved_inputs(wildcards):
    """Polygons of processed countries GISCO does not separate (e.g. Kosovo)."""
    return {
        iso: f"{SPATIAL_DIR}/raw/carved_{iso}.geojson"
        for iso in HD["spatial"]["carved"]
        if iso in hd_countries()
    }


rule prepare_heavy_duty_grid:
    input:
        unpack(carved_inputs),
        countries=rules.download_heavy_duty_spatial_countries.output.countries,
    output:
        raster=f"{SPATIAL_DIR}/grid_countries.tif",
        polygons=f"{SPATIAL_DIR}/grid_countries.parquet",
    log:
        "<logs>/prepare_heavy_duty_grid.log",
    conda:
        "../envs/module.yaml"
    params:
        codes=HD["country_codes"],
        scope=lambda wc: hd_countries(),
        crs=HD["crs"],
        resolution=config["heavy_duty"]["grid"]["resolution_m"],
        bbox_lonlat=HD["spatial"]["bbox_lonlat"],
        buffer_km=HD["spatial"]["buffer_km"],
        carved=HD["spatial"]["carved"],
    message:
        "Allocation grid and country raster (cell centre in the country polygon)."
    script:
        "../scripts/prepare_heavy_duty_grid.py"


rule download_heavy_duty_ghs_pop:
    output:
        zip=f"{SPATIAL_DIR}/raw/ghs_pop_{{epoch}}.zip",
    log:
        "<logs>/download_heavy_duty_ghs_pop_{epoch}.log",
    conda:
        "../envs/module.yaml"
    wildcard_constraints:
        epoch=r"\d{4}",
    params:
        url=lambda wc: HD["spatial"]["ghs_pop"]["url"].format(epoch=wc.epoch),
    message:
        "Download GHS-POP R2023A population, epoch {wildcards.epoch} (1 km, about 320 MB)."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.zip} "{params.url}" 2> {log}'


rule download_heavy_duty_ghs_nres:
    output:
        zip=f"{SPATIAL_DIR}/raw/ghs_built_nres_{{epoch}}.zip",
    log:
        "<logs>/download_heavy_duty_ghs_nres_{epoch}.log",
    conda:
        "../envs/module.yaml"
    wildcard_constraints:
        epoch=r"\d{4}",
    params:
        url=lambda wc: HD["spatial"]["ghs_built_nres"]["url"].format(epoch=wc.epoch),
    message:
        "Download GHSL non-residential built-up surface, epoch {wildcards.epoch} (1 km, about 34 MB)."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.zip} "{params.url}" 2> {log}'


rule prepare_heavy_duty_nres:
    input:
        zip=rules.download_heavy_duty_ghs_nres.output.zip,
        grid=rules.prepare_heavy_duty_grid.output.raster,
        polygons=rules.prepare_heavy_duty_grid.output.polygons,
    output:
        population=f"{SPATIAL_DIR}/non_residential_built_{{epoch}}.tif",
    log:
        "<logs>/prepare_heavy_duty_nres_{epoch}.log",
    conda:
        "../envs/module.yaml"
    wildcard_constraints:
        epoch=r"\d{4}",
    message:
        "Non-residential built-up surface on the allocation grid, epoch {wildcards.epoch}."
    script:
        "../scripts/prepare_heavy_duty_population.py"


def short_haul_weights():
    """Within-region weight rasters of the short-haul key, by epoch (config short_haul.weight)."""
    weight = config["heavy_duty"]["short_haul"]["weight"]
    block, prefix = {
        "population": ("ghs_pop", "population"),
        "non_residential_built": ("ghs_built_nres", "non_residential_built"),
    }[weight]
    return {
        f"weight_{e}": f"{SPATIAL_DIR}/{prefix}_{e}.tif"
        for e in HD["spatial"][block]["epochs"]
    }


def short_haul_epochs():
    weight = config["heavy_duty"]["short_haul"]["weight"]
    block = {"population": "ghs_pop", "non_residential_built": "ghs_built_nres"}[weight]
    return HD["spatial"][block]["epochs"]


rule prepare_heavy_duty_population:
    input:
        zip=rules.download_heavy_duty_ghs_pop.output.zip,
        grid=rules.prepare_heavy_duty_grid.output.raster,
        polygons=rules.prepare_heavy_duty_grid.output.polygons,
    output:
        population=f"{SPATIAL_DIR}/population_{{epoch}}.tif",
    log:
        "<logs>/prepare_heavy_duty_population_{epoch}.log",
    conda:
        "../envs/module.yaml"
    wildcard_constraints:
        epoch=r"\d{4}",
    message:
        "Population on the allocation grid, GHS-POP epoch {wildcards.epoch}."
    script:
        "../scripts/prepare_heavy_duty_population.py"


rule download_heavy_duty_nuts3:
    output:
        nuts=f"{SPATIAL_DIR}/raw/nuts3_{{version}}.geojson",
    log:
        "<logs>/download_heavy_duty_nuts3_{version}.log",
    conda:
        "../envs/module.yaml"
    wildcard_constraints:
        version=r"\d{4}",
    params:
        url=lambda wc: HD["spatial"]["nuts3"]["url"].format(version=wc.version),
    message:
        "Download GISCO NUTS-3 polygons, version {wildcards.version}."
    shell:
        'curl -sSfL --retry 5 --retry-delay 10 -o {output.nuts} "{params.url}" 2> {log}'


rule prepare_heavy_duty_nuts3:
    input:
        **{
            f"nuts_{v}": f"{SPATIAL_DIR}/raw/nuts3_{v}.geojson"
            for v in HD["spatial"]["nuts3"]["versions"]
        },
        grid=rules.prepare_heavy_duty_grid.output.raster,
        loaded=f"{HD_DIR}/eurostat/road_go_na_rl3g.csv",
        unloaded=f"{HD_DIR}/eurostat/road_go_na_ru3g.csv",
    output:
        **{
            f"raster_{v}": f"{SPATIAL_DIR}/nuts3_{v}.tif"
            for v in HD["spatial"]["nuts3"]["versions"]
        },
        codes=f"{SPATIAL_DIR}/nuts3_codes.parquet",
        tonnes=f"{SPATIAL_DIR}/nuts3_tonnes.parquet",
    log:
        "<logs>/prepare_heavy_duty_nuts3.log",
    conda:
        "../envs/module.yaml"
    params:
        versions=HD["spatial"]["nuts3"]["versions"],
    message:
        "NUTS-3 regions on the allocation grid and NUTS-3 tonnes loaded and unloaded."
    script:
        "../scripts/prepare_heavy_duty_nuts3.py"


rule prepare_heavy_duty_etisplus_keys:
    input:
        **short_haul_weights(),
        **{
            f"nuts_{v}": f"{SPATIAL_DIR}/nuts3_{v}.tif"
            for v in HD["spatial"]["nuts3"]["versions"]
        },
        edges=rules.split_etisplus_od_flows.output.edges,
        legs=rules.split_etisplus_od_flows.output.legs,
        nodes=f"{HD_DIR}/etisplus/nodes.csv",
        grid=rules.prepare_heavy_duty_grid.output.raster,
        polygons=rules.prepare_heavy_duty_grid.output.polygons,
        nuts_codes=rules.prepare_heavy_duty_nuts3.output.codes,
        tonnes=rules.prepare_heavy_duty_nuts3.output.tonnes,
    output:
        network_dom=f"{SPATIAL_DIR}/etisplus_network_dom.tif",
        network_int=f"{SPATIAL_DIR}/etisplus_network_int.tif",
        legs_dom=f"{SPATIAL_DIR}/etisplus_legs_dom.tif",
        legs_int=f"{SPATIAL_DIR}/etisplus_legs_int.tif",
    log:
        "<logs>/prepare_heavy_duty_etisplus_keys.log",
    conda:
        "../envs/module.yaml"
    params:
        epochs=short_haul_epochs(),
        versions=HD["spatial"]["nuts3"]["versions"],
        codes=HD["country_codes"],
        reference_year=config["heavy_duty"]["reference_year"],
        ferry_sea_share=HD["etisplus"]["ferry_sea_share"],
    message:
        "ETISplus keys on the grid: E-road vkm and first/last legs, domestic and international."
    script:
        "../scripts/prepare_heavy_duty_etisplus_keys.py"


def spatial_inputs():
    """Inputs shared by the allocation of every year."""
    return {
        **short_haul_weights(),
        **{
            f"nuts_{v}": f"{SPATIAL_DIR}/nuts3_{v}.tif"
            for v in HD["spatial"]["nuts3"]["versions"]
        },
        "grid": rules.prepare_heavy_duty_grid.output.raster,
        "polygons": rules.prepare_heavy_duty_grid.output.polygons,
        "nuts_codes": rules.prepare_heavy_duty_nuts3.output.codes,
        "tonnes": rules.prepare_heavy_duty_nuts3.output.tonnes,
    }


rule allocate_heavy_duty_vkm:
    input:
        **spatial_inputs(),
        levels=rules.fill_heavy_duty_levels.output.levels,
        network_dom=rules.prepare_heavy_duty_etisplus_keys.output.network_dom,
        network_int=rules.prepare_heavy_duty_etisplus_keys.output.network_int,
        legs_dom=rules.prepare_heavy_duty_etisplus_keys.output.legs_dom,
        legs_int=rules.prepare_heavy_duty_etisplus_keys.output.legs_int,
    output:
        international="<results>/heavy_duty/{year}/vkm_international.tif",
        domestic_long="<results>/heavy_duty/{year}/vkm_domestic_long.tif",
        domestic_short="<results>/heavy_duty/{year}/vkm_domestic_short.tif",
        report="<results>/heavy_duty/{year}/allocation_report.parquet",
    log:
        "<logs>/allocate_heavy_duty_vkm_{year}.log",
    conda:
        "../envs/module.yaml"
    wildcard_constraints:
        year=r"\d{4}",
    params:
        epochs=short_haul_epochs(),
        versions=HD["spatial"]["nuts3"]["versions"],
        codes=HD["country_codes"],
    message:
        "Heavy-duty vkm rasters for {wildcards.year}: national levels spread over their keys."
    script:
        "../scripts/allocate_heavy_duty_vkm.py"


rule aggregate_heavy_duty_vkm:
    input:
        **{
            f"{c}_{y}": f"<results>/heavy_duty/{y}/vkm_{c}.tif"
            for y in hd_years()
            for c in ["international", "domestic_long", "domestic_short"]
        },
        shapes="<user_shapes>",
        grid=rules.prepare_heavy_duty_grid.output.raster,
        polygons=rules.prepare_heavy_duty_grid.output.polygons,
    output:
        vkm="<results>/{shapes}/heavy_duty/annual_vkm.parquet",
        diagnostics="<results>/{shapes}/heavy_duty/diagnostics.parquet",
    log:
        "<logs>/{shapes}/aggregate_heavy_duty_vkm.log",
    conda:
        "../envs/module.yaml"
    params:
        years=hd_years(),
    message:
        "Heavy-duty vkm per shape of '{wildcards.shapes}', year and component."
    script:
        "../scripts/aggregate_heavy_duty_vkm.py"
