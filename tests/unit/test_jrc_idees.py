"""Unit tests for the JRC-IDEES 2021 road vehicle-km import."""

import zipfile
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest
import yaml
from combine_national_vkm import combine
from prepare_jrc_idees_road_vkm import (
    CATEGORY_CODES,
    TOTAL_CODE,
    extract_vkm,
    read_road_activity,
    tidy_activity_sheet,
)

MODULE_ROOT = Path(__file__).parents[2]
CC = "XX"
YEARS = [2000, 2001]
# Mvkm per category and year, chosen so they are easy to sum.
VALUES = {
    "heavy_duty": [10.0, 11.0],
    "light_duty": [20.0, 21.0],
    "passenger_car": [300.0, 310.0],
    "motorcycle": [4.0, 5.0],
    "bus": [1.5, 1.6],
}


def make_sheet(cc: str = CC, values: dict | None = None, total_offset: float = 0.0):
    """Build a raw `TrRoad_act`-like sheet with distractor rows, as pandas reads it."""
    values = VALUES if values is None else values
    total = [
        sum(v[i] for v in values.values()) + total_offset for i in range(len(YEARS))
    ]
    rows = [
        ("Transport activity", None, None, None),
        (
            "Passenger transport (mio pkm)",
            999.0,
            999.0,
            f"Activity.Mpkm.{cc}.Tr.Road.Passenger",
        ),
        ("Vehicle-km driven (mio km)", *total, TOTAL_CODE.format(cc=cc)),
        *[(cat, *values[cat], CATEGORY_CODES[cat].format(cc=cc)) for cat in values],
        ("Domestic", 7.0, 7.0, f"VehicleKm.Mkm.{cc}.Tr.Road.Freight.HGV.Domestic"),
        ("Stock of vehicles - in use (vehicles)", 5.0, 5.0, None),
    ]
    return pd.DataFrame(rows, columns=["Road transport - XX", *YEARS, "Code"])


@pytest.fixture
def activity():
    """Tidy activity table built from the synthetic sheet."""
    return tidy_activity_sheet(make_sheet())


def test_tidy_keeps_only_coded_rows_and_int_years(activity):
    """Uncoded rows are dropped and year headers become integers."""
    assert list(activity.columns) == YEARS
    assert activity.index.notna().all()
    assert "Stock of vehicles - in use (vehicles)" not in activity.index
    assert len(activity) == 8


def test_tidy_rejects_sheet_without_code_column():
    """A sheet without the code column is rejected."""
    with pytest.raises(ValueError, match="Code"):
        tidy_activity_sheet(make_sheet().drop(columns="Code"))


def test_tidy_rejects_duplicated_codes():
    """Duplicated codes would make selection ambiguous and are rejected."""
    sheet = make_sheet()
    sheet = pd.concat([sheet, sheet.iloc[[2]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicated"):
        tidy_activity_sheet(sheet)


def test_extract_selects_categories_and_converts_to_vkm(activity):
    """Only the five category rows are kept, converted from Mvkm to vkm."""
    vkm = extract_vkm(activity, CC, "XXX", "test-source")
    assert len(vkm) == len(VALUES) * len(YEARS)
    hdv_2001 = vkm.query("vehicle_category == 'heavy_duty' and year == 2001")["vkm"]
    assert hdv_2001.item() == pytest.approx(11.0e6)
    # HGV "Domestic" row must not leak in.
    assert vkm["vkm"].sum() == pytest.approx(sum(sum(v) for v in VALUES.values()) * 1e6)


def test_extract_labels(activity):
    """Country, unit, source and principle labels are set."""
    vkm = extract_vkm(activity, CC, "XXX", "test-source")
    assert set(vkm["country_id"]) == {"XXX"}
    assert set(vkm["unit"]) == {"vkm"}
    assert set(vkm["source"]) == {"test-source"}
    assert set(vkm["principle"]) == {"unspecified"}
    assert vkm["year"].dtype == "int64"


def test_extract_raises_on_missing_category(activity):
    """A missing category code is an error, not a silent gap."""
    with pytest.raises(KeyError, match="Passenger.Bus"):
        extract_vkm(
            activity.drop(index=CATEGORY_CODES["bus"].format(cc=CC)), CC, "XXX", "s"
        )


def test_extract_raises_on_wrong_country_code(activity):
    """Codes are country-specific; a wrong country code finds nothing."""
    with pytest.raises(KeyError):
        extract_vkm(activity, "YY", "YYY", "s")


def test_extract_raises_when_categories_do_not_sum_to_total():
    """Categories must add up to the total road vehicle-km."""
    activity = tidy_activity_sheet(make_sheet(total_offset=1.0))
    with pytest.raises(ValueError, match="do not sum"):
        extract_vkm(activity, CC, "XXX", "s")


def test_extract_rejects_negative_values():
    """Negative vehicle-km fail schema validation."""
    values = {**VALUES, "bus": [-1.5, 1.6]}
    activity = tidy_activity_sheet(make_sheet(values=values))
    with pytest.raises(pandera.errors.SchemaError):
        extract_vkm(activity, CC, "XXX", "s")


def test_read_road_activity_from_zip(tmp_path):
    """The transport workbook is read from inside a country zip."""
    xlsx = tmp_path / f"JRC-IDEES-2021_Transport_{CC}.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="cover", index=False)
        make_sheet().to_excel(writer, sheet_name="TrRoad_act", index=False)
    zip_path = tmp_path / "country.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(xlsx, xlsx.name)
        zf.writestr(f"JRC-IDEES-2021_Industry_{CC}.xlsx", b"not read")

    activity = read_road_activity(zip_path, CC)
    vkm = extract_vkm(activity, CC, "XXX", "s")
    assert len(vkm) == len(VALUES) * len(YEARS)


def test_combine_sorts_and_rejects_duplicates(activity):
    """Combining sorts by country and rejects duplicated rows."""
    a = extract_vkm(activity, CC, "XXX", "s")
    b = a.assign(country_id="AAA")
    combined = combine([a, b])
    assert list(combined["country_id"].unique()) == ["AAA", "XXX"]
    with pytest.raises(pandera.errors.SchemaError):
        combine([a, a])


def test_settings_country_codes_match_config_schema():
    """Each allowed country has exactly one source; JRC-IDEES covers the EU27."""
    settings = yaml.safe_load(
        (MODULE_ROOT / "workflow/internal/settings.yaml").read_text()
    )
    schema = yaml.safe_load(
        (MODULE_ROOT / "workflow/internal/config.schema.yaml").read_text()
    )
    automatic = settings["resources"]["automatic"]
    jrc = automatic["jrc_idees"]["country_codes"]
    eurostat = automatic["eurostat"]["country_codes"]
    allowed = schema["properties"]["countries"]["items"]["enum"]
    assert not set(jrc) & set(eurostat), "JRC-IDEES has priority: no overlap"
    assert sorted({*jrc, *eurostat}) == sorted(allowed)
    assert len(jrc) == 27
    assert jrc["GRC"] == "EL"
    assert eurostat["NOR"] == "NO"  # not a YAML boolean
    assert set(automatic["eurostat"]["principle"]) == set(eurostat)
