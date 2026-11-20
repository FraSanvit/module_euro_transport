"""Helper functions shared by the module rules."""


def countries_from(source):
    """Configured countries whose national road vehicle-km come from `source`."""
    codes = internal["resources"]["automatic"][source]["country_codes"]
    return [country for country in config["countries"] if country in codes]


def national_source_tables(wildcards):
    """Per-source national vehicle-km tables needed for the configured countries."""
    tables = []
    if countries_from("jrc_idees"):
        tables.append(rules.combine_jrc_idees_road_vkm.output.vkm)
    if countries_from("eurostat"):
        tables.append(rules.prepare_eurostat_road_vkm.output.vkm)
    if countries_from("bfs"):
        tables.append(rules.prepare_bfs_road_vkm.output.vkm)
    return tables


def stock_countries_from(source):
    """Configured countries whose national vehicle stock comes from `source`."""
    block = internal["resources"]["automatic"][source]
    codes = block.get("stock_country_codes", block.get("country_codes"))
    return [country for country in config["countries"] if country in codes]


def national_stock_tables(wildcards):
    """Per-source national vehicle stock tables needed for the configured countries."""
    outputs = {
        "jrc_idees": rules.combine_jrc_idees_road_stock.output.stock,
        "eurostat_stock": rules.prepare_eurostat_road_stock.output.stock,
    }
    return [
        outputs[source]
        for source in internal["resources"]["automatic"]["stock_sources"]
        if stock_countries_from(source)
    ]
