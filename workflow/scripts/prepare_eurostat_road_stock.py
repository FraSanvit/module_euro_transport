"""Derive the road vehicle stock per vehicle category from Eurostat.

Sources (SDMX-CSV, number of vehicles):
- `road_eqs_lorroa` (age = TOTAL): lorries > 3.5 t (LOR_GT3P5), road tractors (TRC)
  and goods vehicles <= 3.5 t (VG_LE3P5).
- `road_eqs_carmot` (mot_nrg = TOTAL, engine = TOTAL): passenger cars.
- `road_eqs_busveh` (vehicle = TOTAL): motor coaches, buses and trolley buses.
- `road_eqs_mopeds` (mot_nrg = TOTAL): motorcycles (MOTO) and mopeds (MOP).

heavy_duty = LOR_GT3P5 + TRC and motorcycle = MOTO + MOP need both parts. Anything
that cannot be derived, or is excluded in the settings, is left out and logged.
"""

import logging

import pandas as pd
from _eurostat import read_sdmx
from _schemas import VEHICLE_CATEGORIES, NationalStockSchema

LOGGER = logging.getLogger(__name__)


def read_stock_tables(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Read and filter the four Eurostat stock datasets."""
    return {
        "lorroa": read_sdmx(
            paths["road_eqs_lorroa"], {"age": "TOTAL", "unit": "NR"}, keep=["vehicle"]
        ),
        "carmot": read_sdmx(
            paths["road_eqs_carmot"],
            {"mot_nrg": "TOTAL", "engine": "TOTAL", "unit": "NR"},
            keep=[],
        ),
        "busveh": read_sdmx(
            paths["road_eqs_busveh"], {"vehicle": "TOTAL", "unit": "NR"}, keep=[]
        ),
        "mopeds": read_sdmx(
            paths["road_eqs_mopeds"],
            {"mot_nrg": "TOTAL", "unit": "NR"},
            keep=["vehicle"],
        ),
    }


def _series(table: pd.DataFrame, vehicle: str | None = None) -> pd.Series:
    if vehicle is not None:
        table = table[table["vehicle"] == vehicle]
    return table.set_index(["geo", "year"])["value"]


def derive_categories(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Map Eurostat vehicle classes to module categories (long `geo, year, category, stock`)."""
    lor, mop = tables["lorroa"], tables["mopeds"]
    parts = {
        # Sums align on (geo, year); a missing part gives NaN, which is dropped.
        "heavy_duty": _series(lor, "LOR_GT3P5") + _series(lor, "TRC"),
        "light_duty": _series(lor, "VG_LE3P5"),
        "passenger_car": _series(tables["carmot"]),
        "bus": _series(tables["busveh"]),
        "motorcycle": _series(mop, "MOTO") + _series(mop, "MOP"),
    }
    frames = [
        series.dropna().rename("stock").reset_index().assign(vehicle_category=category)
        for category, series in parts.items()
    ]
    return pd.concat(frames, ignore_index=True)


def to_national_stock(
    derived: pd.DataFrame,
    geo_codes: dict[str, str],
    exclude: dict[str, list[str]],
    source: str,
) -> pd.DataFrame:
    """Keep the requested countries, drop excluded categories and validate."""
    to_iso = {geo: iso for iso, geo in geo_codes.items()}
    selected = derived[derived["geo"].isin(to_iso)].copy()
    selected["country_id"] = selected["geo"].map(to_iso)
    for country, categories in exclude.items():
        dropped = selected["country_id"].eq(country) & selected[
            "vehicle_category"
        ].isin(categories)
        if dropped.any():
            LOGGER.warning(
                "%s: excluded %s (%d values, see settings).",
                country,
                categories,
                dropped.sum(),
            )
        selected = selected[~dropped]
    for country in sorted(geo_codes):
        group = selected[selected["country_id"] == country]
        for category in VEHICLE_CATEGORIES:
            years = group.loc[group["vehicle_category"] == category, "year"]
            if years.empty:
                LOGGER.warning("%s %s: no Eurostat stock (left missing).", country, category)
            else:
                LOGGER.info(
                    "%s %s: years %s-%s (%d)",
                    country,
                    category,
                    years.min(),
                    years.max(),
                    len(years),
                )
    selected = selected[selected["year"] >= 1990]
    stock = pd.DataFrame(
        {
            "country_id": selected["country_id"],
            "year": selected["year"].astype(int),
            "vehicle_category": selected["vehicle_category"],
            "stock": selected["stock"].astype(float),
            "unit": "vehicles",
            "source": source,
        }
    ).sort_values(["country_id", "year", "vehicle_category"], ignore_index=True)
    return NationalStockSchema.validate(stock)


def main(paths, geo_codes, exclude, source, output_path):
    """Read the datasets, derive categories and write parquet."""
    derived = derive_categories(read_stock_tables(paths))
    to_national_stock(derived, geo_codes, exclude, source).to_parquet(
        output_path, index=False
    )


if __name__ == "__main__":
    logging.basicConfig(
        filename=snakemake.log[0],  # noqa: F821
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    main(
        paths=dict(snakemake.input),  # noqa: F821
        geo_codes=snakemake.params.geo_codes,  # noqa: F821
        exclude=snakemake.params.exclude,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        output_path=snakemake.output.stock,  # noqa: F821
    )
