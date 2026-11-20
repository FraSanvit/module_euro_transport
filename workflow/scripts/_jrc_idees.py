"""Shared helpers to read the JRC-IDEES 2021 road transport sheet.

Rows are selected by their JRC code (last column of `TrRoad_act`), not by position.
"""

import io
import re
import zipfile

import numpy as np
import pandas as pd

SHEET = "TrRoad_act"
CODE_COLUMN = "Code"
SUM_RTOL = 1e-6
# JRC code suffix per module vehicle category, appended to "<prefix>.{cc}.Tr.Road".
CATEGORY_SUFFIXES = {
    "heavy_duty": "Freight.HGV",
    "light_duty": "Freight.LCV",
    "passenger_car": "Passenger.Car",
    "motorcycle": "Passenger.P2W",
    "bus": "Passenger.Bus",
}


def category_codes(prefix: str, jrc_code: str) -> tuple[dict[str, str], str]:
    """Return the JRC codes per category and the total code for a metric prefix.

    Example prefix: "VehicleKm.Mkm" or "Stock.number".
    """
    total = f"{prefix}.{jrc_code}.Tr.Road"
    codes = {cat: f"{total}.{suffix}" for cat, suffix in CATEGORY_SUFFIXES.items()}
    return codes, total


def read_road_activity(zip_path: str, jrc_code: str) -> pd.DataFrame:
    """Read the road activity sheet from a JRC-IDEES country zip.

    Returns a frame indexed by JRC code with one integer column per year.
    """
    member = f"JRC-IDEES-2021_Transport_{jrc_code}.xlsx"
    with zipfile.ZipFile(zip_path) as zf:
        content = zf.read(member)
    sheet = pd.read_excel(io.BytesIO(content), sheet_name=SHEET, header=0)
    return tidy_activity_sheet(sheet)


def tidy_activity_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    """Keep coded rows and year columns of a raw `TrRoad_act` sheet."""
    if CODE_COLUMN not in sheet.columns:
        raise ValueError(f"Sheet has no '{CODE_COLUMN}' column.")
    year_columns = [c for c in sheet.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_columns:
        raise ValueError("Sheet has no year columns.")
    coded = sheet.dropna(subset=[CODE_COLUMN])
    duplicated = coded[CODE_COLUMN][coded[CODE_COLUMN].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Duplicated codes in sheet: {sorted(duplicated.unique())}.")
    table = coded.set_index(CODE_COLUMN)[year_columns]
    table.columns = [int(c) for c in year_columns]
    return table.astype(float)


def select_categories(
    activity: pd.DataFrame, prefix: str, jrc_code: str, country_id: str
) -> pd.DataFrame:
    """Select the five category rows of a metric and return them in long format.

    Returns columns `vehicle_category, year, value` in the sheet's units.
    Raises if a code is missing or if the categories do not add up to the total.
    """
    codes, total_code = category_codes(prefix, jrc_code)
    missing = [c for c in [*codes.values(), total_code] if c not in activity.index]
    if missing:
        raise KeyError(f"Codes missing from JRC-IDEES sheet: {missing}.")

    by_category = activity.loc[list(codes.values())]
    total = activity.loc[total_code]
    if not np.allclose(by_category.sum(), total, rtol=SUM_RTOL, atol=0):
        diff = (by_category.sum() - total).abs().max()
        raise ValueError(
            f"{country_id}: vehicle categories do not sum to {total_code} "
            f"(max abs difference {diff})."
        )

    by_category.index = list(codes.keys())
    long = (
        by_category.rename_axis("vehicle_category")
        .reset_index()
        .melt(id_vars="vehicle_category", var_name="year", value_name="value")
    )
    long["year"] = long["year"].astype(int)
    return long
