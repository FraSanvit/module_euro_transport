"""Annual heavy-duty statistics per country from Eurostat road freight tables (MULTIYEAR_PLAN.md §6.4-§6.7).

Domestic level D = LNAT + ENAT + cabotage of foreign trucks (unreported empty km
estimated from countries with a similar laden trip length, road_go_ta_vm); foreign and own
international activity indices; short-trip share alpha_E. Distrusted survey values
(settings `exclude`) are set missing and logged. Nothing is gap filled.
"""

import logging

import pandas as pd
from _heavy_duty import apply_exclusions, domestic_level, read_eurostat, statistics

LOGGER = logging.getLogger(__name__)


def read_inputs(tott_path, dc_path, hac_path, tra_path, vm_path, ia_paths):
    """Read the Eurostat tables with the filters of MULTIYEAR_PLAN.md §4.3."""
    tott = read_eurostat(tott_path, ["geo", "tra_oper", "unit"], {"tra_type": "TOTAL"})
    dc = read_eurostat(dc_path, ["geo", "distance"], {"tra_type": "TOTAL", "unit": "MIO_VKM"})
    hac = read_eurostat(hac_path, ["geo", "c_cabot"], {"unit": "THS_TKM"})
    tra = read_eurostat(tra_path, ["geo", "c_trans"], {"unit": "THS_T"})
    vm = read_eurostat(vm_path, ["geo", "tra_type", "tra_cov", "loadstat", "unit"])
    ia = pd.concat(
        [read_eurostat(p, ["geo", "c_load", "c_unload"], {"unit": "THS_T"}) for p in ia_paths],
        ignore_index=True,
    )
    return tott, dc, hac, tra, vm, ia


def main(
    paths, ia_paths, codes, years, reference_year, include_cabotage, empty_share,
    cabotage_zero, exclude, correct_coverage, max_coverage_gap, output_path, empty_bandwidth=0.2, distance_break=None,
):  # fmt: skip
    """Compute the statistics for the module's countries and write parquet."""
    tott, dc, hac, tra, vm, ia = read_inputs(
        paths["tott"], paths["dc"], paths["hac"], paths["tra"], paths["vm"], ia_paths
    )
    zero = [codes[c] for c in cabotage_zero if c in codes]
    stats = statistics(
        tott, dc, hac, ia, tra, reference_year, empty_share, zero, correct_coverage,
        max_coverage_gap, vm=vm, empty_bandwidth=empty_bandwidth, distance_break=distance_break,
    )  # fmt: skip
    stats = domestic_level(stats, include_cabotage, zero)
    stats = apply_exclusions(stats, exclude, codes)
    stats["D"] = stats["LNAT"] + stats["ENAT"] + stats["CAB"]
    to_iso = {geo: iso for iso, geo in codes.items()}
    stats = stats[stats["geo"].isin(to_iso) & stats["year"].isin(years)].copy()
    stats.insert(0, "country_id", stats["geo"].map(to_iso))
    usable = set(stats.dropna(subset=["LNAT"])["country_id"])
    for iso in sorted(set(codes) - usable):
        LOGGER.warning("%s: no usable national transport statistic (domestic level left missing).", iso)
    stats.sort_values(["country_id", "year"], ignore_index=True).to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        paths={k: snakemake.input[k] for k in ["tott", "dc", "hac", "tra", "vm"]},  # noqa: F821
        ia_paths=list(snakemake.input.ia),  # noqa: F821
        codes=snakemake.params.codes,  # noqa: F821
        years=snakemake.params.years,  # noqa: F821
        reference_year=snakemake.params.reference_year,  # noqa: F821
        include_cabotage=snakemake.params.include_cabotage,  # noqa: F821
        empty_share=snakemake.params.empty_share,  # noqa: F821
        cabotage_zero=snakemake.params.cabotage_zero,  # noqa: F821
        exclude=snakemake.params.exclude,  # noqa: F821
        correct_coverage=snakemake.params.correct_coverage,  # noqa: F821
        max_coverage_gap=snakemake.params.max_coverage_gap,  # noqa: F821
        empty_bandwidth=snakemake.params.empty_bandwidth,  # noqa: F821
        distance_break=snakemake.params.distance_break,  # noqa: F821
        output_path=snakemake.output.statistics,  # noqa: F821
    )
