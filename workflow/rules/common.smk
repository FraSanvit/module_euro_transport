"""Helper functions shared by the module rules."""


def processed_countries():
    """Configured countries plus the neighbours needed to fill their gaps."""
    neighbours = internal["gap_filling"]["neighbours"]
    extra = [n for country in config["countries"] for n in neighbours.get(country, [])]
    return list(dict.fromkeys(config["countries"] + extra))


def countries_from(source):
    """Processed countries whose national road vehicle-km come from `source`."""
    codes = internal["resources"]["automatic"][source]["country_codes"]
    return [country for country in processed_countries() if country in codes]


def national_source_tables(wildcards):
    """Per-source national vehicle-km tables needed for the processed countries."""
    tables = []
    if countries_from("jrc_idees"):
        tables.append(rules.combine_jrc_idees_road_vkm.output.vkm)
    if countries_from("eurostat"):
        tables.append(rules.prepare_eurostat_road_vkm.output.vkm)
    if countries_from("bfs"):
        tables.append(rules.prepare_bfs_road_vkm.output.vkm)
    if countries_from("makstat"):
        tables.append(rules.prepare_makstat_road_vkm.output.vkm)
    return tables


def pxweb_url(provider, table):
    """PX-Web API URL of a table of a national statistics office."""
    block = internal["resources"]["automatic"][provider]
    return block["url"].format(**block["tables"][table])


def stock_countries_from(source):
    """Processed countries whose national vehicle stock comes from `source`."""
    block = internal["resources"]["automatic"][source]
    codes = block.get("stock_country_codes", block.get("country_codes"))
    return [country for country in processed_countries() if country in codes]


def national_stock_tables(wildcards):
    """Per-source national vehicle stock tables needed for the processed countries."""
    outputs = {
        "jrc_idees": rules.combine_jrc_idees_road_stock.output.stock,
        "eurostat_stock": rules.prepare_eurostat_road_stock.output.stock,
        "bfs": rules.prepare_bfs_road_stock.output.stock,
        "dft": rules.prepare_dft_road_stock.output.stock,
        "makstat": rules.prepare_makstat_road_stock.output.stock,
        "hagstofa": rules.prepare_hagstofa_road_stock.output.stock,
        "ask": rules.prepare_ask_road_stock.output.stock,
    }
    return [
        outputs[source]
        for source in internal["resources"]["automatic"]["stock_sources"]
        if stock_countries_from(source)
    ]


# Heavy-duty vehicle-km (MULTIYEAR_PLAN.md).
HD = internal["heavy_duty"]
HD_DIR = "<resources>/automatic/heavy_duty"


def hd_countries():
    """Processed countries (configured + gap-filling neighbours) with a heavy-duty code."""
    return [c for c in processed_countries() if c in HD["country_codes"]]


def hd_years():
    return config["heavy_duty"]["years"]


def trafa_url():
    years = range(HD["territorial_totals"]["trafa"]["first_year"], max(hd_years()) + 1)
    return HD["territorial_totals"]["trafa"]["url"].format(
        years=",".join(str(y) for y in years)
    )


def hd_totals_inputs(wildcards):
    """Territorial-total sources of the processed override countries."""
    countries = hd_countries()
    inputs = {
        "dft": f"{HD_DIR}/dft_region_traffic.csv",
        "uk_population": f"{HD_DIR}/eurostat/demo_r_pjanaggr3_uk.csv",
        "trafa": f"{HD_DIR}/trafa_t04010.json",
    }
    if "CHE" in countries:
        inputs["bfs"] = rules.prepare_bfs_road_vkm.output.vkm
    return inputs


def hd_downloads():
    """Downloaded file -> URL, for the manifest."""
    files = {
        f"{HD_DIR}/eurostat/{name}.csv": HD["eurostat"]["url"].format(query=query)
        for name, query in HD["eurostat"]["queries"].items()
    }
    for year in hd_years():
        query = HD["eurostat"]["road_go_ia_rc"].format(year=year)
        files[f"{HD_DIR}/eurostat/road_go_ia_rc_{year}.csv"] = HD["eurostat"][
            "url"
        ].format(query=query)
    for key, file in HD["etisplus"]["files"].items():
        files[f"{HD_DIR}/etisplus/{key}.csv"] = HD["etisplus"]["url"].format(
            file_id=file["file_id"]
        )
    files[f"{HD_DIR}/gisco/countries.geojson"] = HD["gisco"]["countries"]
    files[f"{HD_DIR}/dft_region_traffic.csv"] = HD["territorial_totals"]["dft"]["url"]
    files[f"{HD_DIR}/trafa_t04010.json"] = trafa_url()
    if "DEU" in hd_countries():
        files[f"{HD_DIR}/zenodo/national_data.zip"] = HD["zenodo"]["url"]
    return files
