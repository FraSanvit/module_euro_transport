"""Helpers shared by the BFS (Swiss Federal Statistical Office) Excel imports."""

import re
import unicodedata

import pandas as pd

HEADER_SEARCH_ROWS = 10


def normalise_label(label) -> str:
    """Normalise a row label: collapse whitespace and drop trailing footnote markers.

    E.g. 'Leichte Fahrzeuge3', 'Autobusse 3)' and 'Strasse 2) 7)' become
    'Leichte Fahrzeuge', 'Autobusse' and 'Strasse'. Non-strings give ''.
    """
    if not isinstance(label, str):
        return ""
    text = unicodedata.normalize("NFC", label)
    text = " ".join(text.split())
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\s*\d+\)?$", "", text).strip()
    return text


def parse_year(value) -> int | None:
    """Return the year in a header cell, or None if the cell is not a year."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value) or not float(value).is_integer():
            return None
        year = int(value)
    elif isinstance(value, str) and re.fullmatch(r"\s*\d{4}\s*", value):
        year = int(value)
    else:
        return None
    return year if 1900 <= year <= 2100 else None


def find_year_columns(sheet: pd.DataFrame) -> tuple[int, dict[int, int]]:
    """Locate the header row and map column positions to years.

    The header row is the first of the top rows with an empty label column and at
    least one year cell. Columns without a year (footnote flags) are not returned.
    """
    for row in range(min(HEADER_SEARCH_ROWS, len(sheet))):
        years = {
            col: parse_year(sheet.iat[row, col]) for col in range(1, sheet.shape[1])
        }
        years = {col: year for col, year in years.items() if year is not None}
        if years and normalise_label(sheet.iat[row, 0]) == "":
            if len(set(years.values())) != len(years):
                raise ValueError(f"Duplicated years in header row {row}.")
            return row, years
    raise ValueError("No header row with years found.")
