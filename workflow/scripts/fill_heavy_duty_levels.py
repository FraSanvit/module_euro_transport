"""Fill the heavy-duty national levels for every country and year (MULTIYEAR_PLAN.md §6.9).

Domestic level D, international level L_int and long-haul share alpha_E: own series
first (ratios interpolated, or extended with a clipped Theil-Sen trend), then
neighbours (D per GDP in PPS, L_int per D, mean alpha_E). L_long = alpha_E x D,
L_short = D - L_long, total = D + L_int. Every filled value is labelled and logged.
"""

import logging

import pandas as pd
from _heavy_duty import read_eurostat
from _heavy_duty_fill import fill_levels
from _schemas import HeavyDutyFilledSchema

LOGGER = logging.getLogger(__name__)
COLUMNS = [
    "country_id", "year", "L_int", "L_int_foreign", "L_int_own", "CAB", "L_dom", "L_long", "L_short",
    "total", "alpha_E", "T", "unit", "source", "flags", "filled_L_dom", "filled_L_int", "filled_alpha_E",
]  # fmt: skip
GDP_UNITS = {"CP_MPPS_EU27_2020": "gdp_pps", "CP_MEUR": "gdp_eur"}


def read_gdp(path, codes: dict) -> pd.DataFrame:
    """GDP (B1GQ) in PPS and in euro per ISO3 country: `country_id, year, gdp_pps, gdp_eur`."""
    raw = read_eurostat(path, ["geo", "unit"], {"na_item": "B1GQ"})
    raw = raw[raw["unit"].isin(GDP_UNITS)]
    to_iso = {geo: iso for iso, geo in codes.items()}
    raw = raw[raw["geo"].isin(to_iso)].assign(country_id=lambda t: t["geo"].map(to_iso))
    wide = raw.pivot_table(index=["country_id", "year"], columns="unit", values="value").rename(columns=GDP_UNITS)
    return wide.reindex(columns=list(GDP_UNITS.values())).reset_index()


def main(
    levels_path, gdp_path, network_path, codes, years, neighbours, max_years, settings, reference_year,
    islands, output_path,
):  # fmt: skip
    """Fill the levels and write parquet."""
    levels = pd.read_parquet(levels_path)
    gdp = read_gdp(gdp_path, codes)
    out = fill_levels(
        levels, gdp, years, neighbours, max_years, settings["trend_years"],
        settings["outlier_mad"], settings["outlier_min_abs"], pd.read_parquet(network_path),
        reference_year, islands, settings["etisplus_min_coverage"],
    )  # fmt: skip
    out["unit"] = "vkm"
    out = out[COLUMNS].sort_values(["country_id", "year"], ignore_index=True)
    for name in ["filled_L_dom", "filled_L_int", "filled_alpha_E"]:
        counts = out[name].value_counts().to_dict()
        LOGGER.info("%s: %s", name, counts)
    HeavyDutyFilledSchema.validate(out).to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        levels_path=snakemake.input.levels,  # noqa: F821
        gdp_path=snakemake.input.gdp,  # noqa: F821
        network_path=snakemake.input.network,  # noqa: F821
        codes=snakemake.params.codes,  # noqa: F821
        years=snakemake.params.years,  # noqa: F821
        neighbours=snakemake.params.neighbours,  # noqa: F821
        max_years=snakemake.params.max_years,  # noqa: F821
        settings=snakemake.params.settings,  # noqa: F821
        reference_year=snakemake.params.reference_year,  # noqa: F821
        islands=snakemake.params.islands,  # noqa: F821
        output_path=snakemake.output.levels,  # noqa: F821
    )
