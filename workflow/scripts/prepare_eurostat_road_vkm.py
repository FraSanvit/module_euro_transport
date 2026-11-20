"""Derive annual road vehicle-km per vehicle category from Eurostat.

Sources (SDMX-CSV, million vehicle-km):
- `road_tf_vehmov`, traffic on the national territory by domestic and foreign
  vehicles (`regisveh = TERNAT_REG`): passenger cars, buses, motorcycles.
- `road_tf_road`, all road types (`tra_infr = TOTAL`): lorries by weight class and
  road tractors, used for the heavy/light split.

Categories that cannot be derived for a country-year are left out and logged.
No values are estimated or filled. Countries listed in `categories` keep only those
categories (the others come from another source).
"""

import logging
import sys

import pandas as pd
from _eurostat import read_sdmx
from _schemas import VEHICLE_CATEGORIES, NationalVkmSchema

LOGGER = logging.getLogger(__name__)
MKM_TO_KM = 1e6
VEHMOV_CATEGORIES = {
    "passenger_car": "CAR",
    "bus": "BUS_MCO_TRO",
    "motorcycle": "MOTO_MOP",
}
LORRY_CHECK_RTOL = 0.01


def read_sdmx_csv(path, dimension: str, value: str) -> pd.DataFrame:
    """Read a Eurostat SDMX-CSV file and keep one value of a filter dimension.

    Returns columns `geo, vehicle, year, mvkm, flag`.
    """
    table = read_sdmx(path, {dimension: value, "unit": "MIO_VKM"}, keep=["vehicle"])
    return table.rename(columns={"value": "mvkm"})


def _wide(table: pd.DataFrame) -> pd.DataFrame:
    duplicated = table.duplicated(["geo", "year", "vehicle"])
    if duplicated.any():
        raise ValueError("Duplicated Eurostat observations.")
    return table.pivot(index=["geo", "year"], columns="vehicle", values="mvkm")


def derive_categories(vehmov: pd.DataFrame, road: pd.DataFrame) -> pd.DataFrame:
    """Map Eurostat vehicle classes to module categories.

    - passenger_car, bus, motorcycle: `road_tf_vehmov` CAR, BUS_MCO_TRO, MOTO_MOP.
    - light_duty: `road_tf_road` lorries <= 3.5 t (LOR_LE3P5).
    - heavy_duty: `road_tf_road` lorries (LOR) - lorries <= 3.5 t + road tractors (TRC);
      only where all three are reported.

    Returns long columns `geo, year, vehicle_category, mvkm`.
    """
    v = _wide(vehmov)
    r = _wide(road)
    frames = []
    for category, code in VEHMOV_CATEGORIES.items():
        if code in v:
            frames.append(v[code].rename(category))
    if "LOR_LE3P5" in r:
        frames.append(r["LOR_LE3P5"].rename("light_duty"))
    if {"LOR", "LOR_LE3P5", "TRC"} <= set(r.columns):
        heavy = r["LOR"] - r["LOR_LE3P5"] + r["TRC"]
        if (heavy < 0).any():
            bad = heavy[heavy < 0].index.tolist()
            raise ValueError(f"Negative heavy-duty vehicle-km derived for {bad}.")
        frames.append(heavy.rename("heavy_duty"))
        _check_lorry_totals(v, r)
    if not frames:
        return pd.DataFrame(columns=["geo", "year", "vehicle_category", "mvkm"])
    wide = pd.concat(frames, axis=1)
    long = wide.rename_axis(columns="vehicle_category").stack().dropna().rename("mvkm")
    return long.reset_index()


def _check_lorry_totals(v: pd.DataFrame, r: pd.DataFrame) -> None:
    """Warn where road_tf_road LOR + TRC disagrees with road_tf_vehmov LOR_RTRN."""
    if "LOR_RTRN" not in v:
        return
    both = pd.concat([(r["LOR"] + r["TRC"]).rename("road"), v["LOR_RTRN"]], axis=1)
    both = both.dropna()
    off = (both["road"] / both["LOR_RTRN"] - 1).abs() > LORRY_CHECK_RTOL
    for (geo, year), row in both[off].iterrows():
        LOGGER.warning(
            "%s %s: road_tf_road LOR+TRC = %.0f differs from road_tf_vehmov "
            "LOR_RTRN = %.0f Mvkm.",
            geo,
            year,
            row["road"],
            row["LOR_RTRN"],
        )


def to_national_vkm(
    derived: pd.DataFrame,
    country_codes: dict[str, str],
    principles: dict[str, str],
    source: str,
    categories: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Select the requested countries and convert to the national vehicle-km schema.

    `country_codes` maps ISO alpha-3 -> Eurostat geo code for the countries to keep.
    `categories` maps ISO alpha-3 -> vehicle categories to keep (default: all).
    Raises if a requested country has no data at all.
    """
    categories = categories or {}
    to_iso = {geo: iso for iso, geo in country_codes.items()}
    selected = derived[derived["geo"].isin(to_iso)].copy()
    empty = sorted(set(country_codes) - set(selected["geo"].map(to_iso)))
    if empty:
        raise ValueError(f"No Eurostat road vehicle-km for {empty}.")
    selected["country_id"] = selected["geo"].map(to_iso)
    allowed = selected["country_id"].map(
        lambda country: categories.get(country, VEHICLE_CATEGORIES)
    )
    keep = [cat in cats for cat, cats in zip(selected["vehicle_category"], allowed)]
    selected = selected[keep]
    for country, group in selected.groupby("country_id"):
        for category, cat_group in group.groupby("vehicle_category"):
            LOGGER.info(
                "%s %s: years %s-%s (%d)",
                country,
                category,
                cat_group["year"].min(),
                cat_group["year"].max(),
                len(cat_group),
            )
        wanted = categories.get(country, VEHICLE_CATEGORIES)
        missing = sorted(set(wanted) - set(group["vehicle_category"]))
        if missing:
            LOGGER.warning("%s: no Eurostat data for %s (left missing).", country, missing)
    national = pd.DataFrame(
        {
            "country_id": selected["country_id"],
            "year": selected["year"].astype(int),
            "vehicle_category": selected["vehicle_category"],
            "vkm": selected["mvkm"].astype(float) * MKM_TO_KM,
            "unit": "vkm",
            "source": source,
            "principle": selected["country_id"].map(principles),
        }
    )
    if national["principle"].isna().any():
        raise ValueError("A principle must be set for every Eurostat country.")
    national = national.sort_values(
        ["country_id", "year", "vehicle_category"], ignore_index=True
    )
    return NationalVkmSchema.validate(national)


def log_flags(table: pd.DataFrame, geos: list[str], dataset: str) -> None:
    """Log how many flagged observations (estimates, breaks, ...) are used."""
    flagged = table[table["geo"].isin(geos) & (table["flag"].fillna("") != "")]
    for (geo, flag), group in flagged.groupby(["geo", "flag"]):
        LOGGER.info("%s %s: %d values flagged '%s'", dataset, geo, len(group), flag)


def main(
    vehmov_path, road_path, country_codes, principles, categories, source, output_path
):
    """Read both datasets, derive categories and write parquet."""
    vehmov = read_sdmx_csv(vehmov_path, "regisveh", "TERNAT_REG")
    road = read_sdmx_csv(road_path, "tra_infr", "TOTAL")
    geos = list(country_codes.values())
    vehmov = vehmov[vehmov["geo"].isin(geos)]
    road = road[road["geo"].isin(geos)]
    log_flags(vehmov, geos, "road_tf_vehmov")
    log_flags(road, geos, "road_tf_road")
    derived = derive_categories(vehmov, road)
    national = to_national_vkm(derived, country_codes, principles, source, categories)
    national.to_parquet(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        vehmov_path=snakemake.input.road_tf_vehmov,  # noqa: F821
        road_path=snakemake.input.road_tf_road,  # noqa: F821
        country_codes=snakemake.params.country_codes,  # noqa: F821
        principles=snakemake.params.principles,  # noqa: F821
        categories=snakemake.params.categories,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        output_path=snakemake.output.vkm,  # noqa: F821
    )
