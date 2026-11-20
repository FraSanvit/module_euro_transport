"""Shared helpers to read MAKSTAT (North Macedonia) PX-Web downloads."""

import json
import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)
MISSING = {"..", "...", "-", ""}
"""PX-Web symbols for values that are not available."""


def read_px_csv(path) -> pd.DataFrame:
    """Read a PX-Web CSV export (first column = row labels) as floats.

    Returns the table indexed by the stripped row label; missing symbols become NaN.
    Raises on duplicated row labels.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    labels = raw.iloc[:, 0].str.strip()
    if labels.duplicated().any():
        raise ValueError(f"{path}: duplicated rows {labels[labels.duplicated()].tolist()}.")
    values = raw.iloc[:, 1:].apply(lambda col: col.str.strip())
    values = values.mask(values.isin(MISSING)).astype(float)
    values.index = labels
    return values


def log_updated(listing_path, table_id: str) -> str:
    """Log the "updated" date of a table from its PX-Web folder listing."""
    with open(listing_path, encoding="utf-8") as f:
        listing = json.load(f)
    entries = [entry for entry in listing if entry.get("id") == table_id]
    if len(entries) != 1:
        raise KeyError(f"{listing_path}: table {table_id} not listed.")
    updated = entries[0].get("updated", "unknown")
    LOGGER.info("MAKSTAT %s: last updated %s", table_id, updated)
    return updated
