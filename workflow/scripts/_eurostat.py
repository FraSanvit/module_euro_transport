"""Shared helpers to read Eurostat SDMX-CSV downloads."""

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def read_sdmx(path, filters: dict[str, str], keep: list[str]) -> pd.DataFrame:
    """Read a Eurostat SDMX-CSV file, filter dimensions and drop empty values.

    `filters` maps dimension -> value to keep; `keep` lists the dimensions to return
    besides `geo`. Returns columns `geo, *keep, year, value, flag`.
    The dataset's "LAST UPDATE" is logged, since the API is not versioned.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {*filters, *keep, "geo", "TIME_PERIOD", "OBS_VALUE"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}.")
    for update in sorted(raw.get("LAST UPDATE", pd.Series(dtype=str)).unique()):
        LOGGER.info("%s: Eurostat last update %s", path, update)
    for dimension, value in filters.items():
        raw = raw[raw[dimension] == value]
    raw = raw[raw["OBS_VALUE"].str.strip() != ""]
    table = raw[["geo", *keep]].copy()
    table["year"] = raw["TIME_PERIOD"].astype(int)
    table["value"] = raw["OBS_VALUE"].astype(float)
    table["flag"] = raw["OBS_FLAG"] if "OBS_FLAG" in raw else ""
    table = table.reset_index(drop=True)
    if table.duplicated(["geo", *keep, "year"]).any():
        raise ValueError(f"{path}: duplicated observations after filtering {filters}.")
    return table
