"""Shared helpers for national tables (vehicle-km or vehicle stock)."""

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)
ALL_YEARS = "all"


def drop_excluded(table: pd.DataFrame, exclude: dict) -> pd.DataFrame:
    """Drop country-category-years known to be unreliable (see settings `exclude`).

    `exclude` maps country -> category -> list of years, or "all". Every dropped
    combination is logged; nothing is replaced.
    """
    keep = pd.Series(True, index=table.index)
    for country, categories in (exclude or {}).items():
        for category, years in categories.items():
            hit = (table["country_id"] == country) & (
                table["vehicle_category"] == category
            )
            if years != ALL_YEARS:
                hit &= table["year"].isin(years)
            if hit.any():
                LOGGER.warning(
                    "%s %s: years %s excluded as unreliable (left missing).",
                    country,
                    category,
                    sorted(table.loc[hit, "year"].tolist()),
                )
            keep &= ~hit
    return table[keep].reset_index(drop=True)
