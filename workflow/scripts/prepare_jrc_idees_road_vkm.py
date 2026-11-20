"""Extract annual road vehicle-km per vehicle category from a JRC-IDEES 2021 country zip.

Reads sheet `TrRoad_act` of `JRC-IDEES-2021_Transport_{CC}.xlsx` and selects rows by
their JRC code (last column), not by position.
"""

import io
import re
import sys
import zipfile

import numpy as np
import pandas as pd
from _schemas import NationalVkmSchema

SHEET = "TrRoad_act"
CODE_COLUMN = "Code"
MKM_TO_KM = 1e6
TOTAL_CODE = "VehicleKm.Mkm.{cc}.Tr.Road"
CATEGORY_CODES = {
    "heavy_duty": "VehicleKm.Mkm.{cc}.Tr.Road.Freight.HGV",
    "light_duty": "VehicleKm.Mkm.{cc}.Tr.Road.Freight.LCV",
    "passenger_car": "VehicleKm.Mkm.{cc}.Tr.Road.Passenger.Car",
    "motorcycle": "VehicleKm.Mkm.{cc}.Tr.Road.Passenger.P2W",
    "bus": "VehicleKm.Mkm.{cc}.Tr.Road.Passenger.Bus",
}
SUM_RTOL = 1e-6


def read_road_activity(zip_path: str, jrc_code: str) -> pd.DataFrame:
    """Read the road activity sheet from a JRC-IDEES country zip.

    Returns a frame indexed by JRC code with one integer column per year,
    values in million vehicle-km for the vehicle-km rows.
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


def extract_vkm(
    activity: pd.DataFrame, jrc_code: str, country_id: str, source: str
) -> pd.DataFrame:
    """Select the vehicle-km rows per category and return them in long format.

    Raises if a category code is missing or if the categories do not add up to
    the total road vehicle-km.
    """
    codes = {cat: code.format(cc=jrc_code) for cat, code in CATEGORY_CODES.items()}
    total_code = TOTAL_CODE.format(cc=jrc_code)
    missing = [c for c in [*codes.values(), total_code] if c not in activity.index]
    if missing:
        raise KeyError(f"Codes missing from JRC-IDEES sheet: {missing}.")

    by_category = activity.loc[list(codes.values())]
    total = activity.loc[total_code]
    if not np.allclose(by_category.sum(), total, rtol=SUM_RTOL, atol=0):
        diff = (by_category.sum() - total).abs().max()
        raise ValueError(
            f"{country_id}: vehicle categories do not sum to total road vkm "
            f"(max abs difference {diff} Mvkm)."
        )

    by_category.index = list(codes.keys())
    long = (
        by_category.rename_axis("vehicle_category")
        .reset_index()
        .melt(id_vars="vehicle_category", var_name="year", value_name="vkm")
    )
    long["vkm"] = long["vkm"] * MKM_TO_KM
    long["year"] = long["year"].astype(int)
    long["country_id"] = country_id
    long["unit"] = "vkm"
    long["source"] = source
    long["principle"] = "unspecified"
    long = long[list(NationalVkmSchema.columns)]
    long = long.sort_values(["year", "vehicle_category"], ignore_index=True)
    return NationalVkmSchema.validate(long)


def main(zip_path, jrc_code, country_id, source, output_path):
    """Extract one country and write it as parquet."""
    activity = read_road_activity(zip_path, jrc_code)
    vkm = extract_vkm(activity, jrc_code, country_id, source)
    vkm.to_parquet(output_path, index=False)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")  # noqa: F821
    main(
        zip_path=snakemake.input.zip,  # noqa: F821
        jrc_code=snakemake.params.jrc_code,  # noqa: F821
        country_id=snakemake.wildcards.country,  # noqa: F821
        source=snakemake.params.source,  # noqa: F821
        output_path=snakemake.output.vkm,  # noqa: F821
    )
