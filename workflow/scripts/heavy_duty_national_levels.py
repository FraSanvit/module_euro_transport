"""Heavy-duty national levels per country and year, before gap filling (MULTIYEAR_PLAN.md §6.3-§6.8).

1. Reference-year calibration (§6.3): international level from the ETISplus network
   (k x ETIS_int; 0 for islands; fitted into the territorial total where one exists),
   foreign/own split by tonnes (w), vkm per tonne of the foreign-hauliers index (r_f),
   share of own hauliers' international vkm driven in the country (r_o), long-haul
   share of domestic vkm (s), toll scaling (gamma, toll countries).
2. Each year: international = foreign (toll, else r_f x corrected index) + own
   (r_o x own international vkm); domestic D from the statistics, split into long-
   and short-haul (§6.7); territorial totals fix the total where they exist (§6.8).

Values stay missing where an input is missing; they are filled in the common
gap-filling step. Every adjustment is flagged.
"""

import logging

import numpy as np
import pandas as pd
from _heavy_duty import annual_levels, calibrate
from _schemas import HeavyDutyCalibrationSchema, HeavyDutyLevelsSchema

LOGGER = logging.getLogger(__name__)
LEVEL_COLUMNS = [
    "LNAT", "ENAT", "CAB", "D", "I_f", "coverage_m", "I_f_corrected", "J_o", "I_o",
    "alpha_E", "s_long", "L_int_foreign", "L_int_own", "L_int", "L_dom", "L_long",
    "L_short", "total", "T",
]  # fmt: skip


def with_geo(table: pd.DataFrame, codes: dict) -> pd.DataFrame:
    """Add the Eurostat code `geo` from `country_id`."""
    return table.assign(geo=table["country_id"].map(codes))


def to_output(levels: pd.DataFrame, codes: dict, totals: pd.DataFrame) -> pd.DataFrame:
    """ISO3 ids, units and sources; validate."""
    to_iso = {geo: iso for iso, geo in codes.items()}
    out = levels.copy()
    out.insert(0, "country_id", out.pop("geo").map(to_iso))
    source = totals.set_index(["country_id", "year"])["source"]
    keys = pd.MultiIndex.from_frame(out[["country_id", "year"]])
    total_source = ("territorial total: " + source.reindex(keys).fillna("").astype(str)).to_numpy()
    level_source = out.pop("level_source").to_numpy()
    out["source"] = np.where(level_source == "territorial total", total_source, level_source)
    out["unit"] = "vkm"
    out = out[["country_id", "year", *LEVEL_COLUMNS, "unit", "source", "flags"]]
    return HeavyDutyLevelsSchema.validate(out.sort_values(["country_id", "year"], ignore_index=True))


def main(
    statistics_path, network_path, totals_path, toll_path, codes, countries, years,
    reference_year, parameters, islands, toll_countries, toll_cabotage_share, jump,
    levels_path, calibration_path,
):  # fmt: skip
    """Calibrate at the reference year, compute annual levels and write parquet."""
    if reference_year not in years:
        raise ValueError(f"Reference year {reference_year} is not in years {years}.")
    codes = {c: codes[c] for c in countries}
    stats = pd.read_parquet(statistics_path).drop(columns="country_id")
    network = with_geo(pd.read_parquet(network_path), codes).dropna(subset=["geo"])
    totals = with_geo(pd.read_parquet(totals_path), codes)
    toll = with_geo(pd.read_parquet(toll_path), codes) if toll_path else None
    geo_islands = [codes[c] for c in islands if c in codes]
    geo_toll = [codes[c] for c in toll_countries if c in codes]
    k, f_net = parameters["k"], parameters["f_net"]
    calibration = calibrate(
        stats, network, totals, toll, reference_year, k,
        geo_islands, geo_toll, toll_cabotage_share, countries=list(codes.values()),
    )  # fmt: skip
    levels = annual_levels(
        stats, calibration, totals, toll, years, f_net, geo_islands, geo_toll,
        toll_cabotage_share, jump,
    )  # fmt: skip
    levels = to_output(levels, codes, totals)
    for country, row in levels[levels["year"] == reference_year].set_index("country_id").iterrows():
        LOGGER.info(
            "%s %d: total %.4g vkm (international %.4g, domestic %.4g) [%s] %s",
            country, reference_year, row["total"], row["L_int"], row["L_dom"], row["source"], row["flags"],
        )  # fmt: skip
    to_iso = {geo: iso for iso, geo in codes.items()}
    calibration.insert(0, "country_id", calibration.pop("geo").map(to_iso))
    calibration.insert(1, "reference_year", reference_year)
    HeavyDutyCalibrationSchema.validate(calibration.sort_values("country_id", ignore_index=True)).to_parquet(
        calibration_path, index=False
    )
    levels.to_parquet(levels_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        statistics_path=snakemake.input.statistics,  # noqa: F821
        network_path=snakemake.input.network,  # noqa: F821
        totals_path=snakemake.input.totals,  # noqa: F821
        toll_path=snakemake.input.get("toll"),  # noqa: F821
        codes=snakemake.params.codes,  # noqa: F821
        countries=snakemake.params.countries,  # noqa: F821
        years=snakemake.params.years,  # noqa: F821
        reference_year=snakemake.params.reference_year,  # noqa: F821
        parameters=snakemake.params.parameters,  # noqa: F821
        islands=snakemake.params.islands,  # noqa: F821
        toll_countries=snakemake.params.toll_countries,  # noqa: F821
        toll_cabotage_share=snakemake.params.toll_cabotage_share,  # noqa: F821
        jump=snakemake.params.jump,  # noqa: F821
        levels_path=snakemake.output.levels,  # noqa: F821
        calibration_path=snakemake.output.calibration,  # noqa: F821
    )
