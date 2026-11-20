"""Unit tests for the JRC-IDEES 2021 road vehicle-km and vehicle stock import."""

import zipfile
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest
import yaml
from _jrc_idees import category_codes
from _schemas import NationalStockSchema
from combine_national_tables import combine, log_missing_countries
from prepare_jrc_idees_road_stock import extract_stock
from prepare_jrc_idees_road_vkm import extract_vkm, read_road_activity, tidy_activity_sheet

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
# Number of vehicles per category and year.
STOCK = {
    "heavy_duty": [100.0, 110.0],
    "light_duty": [400.0, 410.0],
    "passenger_car": [5000.0, 5100.0],
    "motorcycle": [600.0, 610.0],
    "bus": [10.0, 11.0],
}
VKM, STK = "VehicleKm.Mkm", "Stock.number"


def _block(prefix, cc, values, total_offset=0.0):
    codes, total_code = category_codes(prefix, cc)
    total = [
        sum(v[i] for v in values.values()) + total_offset for i in range(len(YEARS))
    ]
    return [
        (f"{prefix} total", *total, total_code),
        *[(cat, *values[cat], codes[cat]) for cat in values],
    ]


def make_sheet(
    cc: str = CC,
    values: dict | None = None,
    total_offset: float = 0.0,
    stock: dict | None = None,
    stock_offset: float = 0.0,
):
    """Build a raw `TrRoad_act`-like sheet with distractor rows, as pandas reads it."""
    values = VALUES if values is None else values
    stock = STOCK if stock is None else stock
    rows = [
        ("Transport activity", None, None, None),
        (
            "Passenger transport (mio pkm)",
            999.0,
            999.0,
            f"Activity.Mpkm.{cc}.Tr.Road.Passenger",
        ),
        *_block(VKM, cc, values, total_offset),
        ("Domestic", 7.0, 7.0, f"VehicleKm.Mkm.{cc}.Tr.Road.Freight.HGV.Domestic"),
        *_block(STK, cc, stock, stock_offset),
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
    assert len(activity) == 14


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
            activity.drop(index=category_codes(VKM, CC)[0]["bus"]), CC, "XXX", "s"
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
        (MODULE_ROOT / "workflow/internal/settings.yaml").read_text(encoding="utf-8")
    )
    schema = yaml.safe_load(
        (MODULE_ROOT / "workflow/internal/config.schema.yaml").read_text()
    )
    automatic = settings["resources"]["automatic"]
    routed = {
        source: set(block["country_codes"])
        for source, block in automatic.items()
        if isinstance(block, dict) and "country_codes" in block
    }
    allowed = schema["properties"]["countries"]["items"]["enum"]
    sources = sorted(routed)
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            assert not routed[a] & routed[b], f"{a} and {b} overlap"
    assert sorted(set().union(*routed.values())) == sorted(allowed)
    for source, block in automatic.items():
        if isinstance(block, dict) and "principle" in block:
            assert set(block["principle"]) == routed[source], source
    jrc = automatic["jrc_idees"]["country_codes"]
    assert len(jrc) == 27
    assert jrc["GRC"] == "EL"
    assert automatic["eurostat"]["country_codes"]["NOR"] == "NO"  # not a YAML boolean
    # Vehicle stock: Eurostat geo codes for every allowed country; priority sources exist.
    assert set(allowed) <= set(automatic["eurostat_stock"]["geo_codes"])
    assert automatic["eurostat_stock"]["geo_codes"]["NOR"] == "NO"
    for source in automatic["stock_priority"]:
        assert "source" in automatic[source], source


def test_extract_stock_selects_categories(activity):
    """Stock rows are selected per category, in number of vehicles."""
    stock = extract_stock(activity, CC, "XXX", "test-source")
    assert len(stock) == len(STOCK) * len(YEARS)
    hdv = stock.query("vehicle_category == 'heavy_duty' and year == 2001")["stock"]
    assert hdv.item() == 110.0
    assert stock["stock"].sum() == pytest.approx(sum(sum(v) for v in STOCK.values()))
    assert set(stock["unit"]) == {"vehicles"}
    assert list(stock.columns) == list(NationalStockSchema.columns)


def test_extract_stock_ignores_vkm_rows(activity):
    """Stock and vehicle-km rows do not mix (values differ by construction)."""
    stock = extract_stock(activity, CC, "XXX", "s")
    vkm = extract_vkm(activity, CC, "XXX", "s")
    assert stock["stock"].sum() != pytest.approx(vkm["vkm"].sum() / 1e6)


def test_extract_stock_raises_when_categories_do_not_sum_to_total():
    """Stock categories must add up to the total road vehicle stock."""
    activity = tidy_activity_sheet(make_sheet(stock_offset=3.0))
    with pytest.raises(ValueError, match="Stock.number"):
        extract_stock(activity, CC, "XXX", "s")


def test_extract_stock_raises_on_missing_category(activity):
    """A missing stock category code is an error."""
    code = category_codes(STK, CC)[0]["motorcycle"]
    with pytest.raises(KeyError, match="Passenger.P2W"):
        extract_stock(activity.drop(index=code), CC, "XXX", "s")


def test_combine_stock_uses_stock_schema(activity):
    """Combining stock tables validates them with the stock schema."""
    stock = extract_stock(activity, CC, "XXX", "s")
    combined = combine([stock, stock.assign(country_id="AAA")], NationalStockSchema)
    assert list(combined["country_id"].unique()) == ["AAA", "XXX"]
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        combine([stock])  # default schema is the vehicle-km schema


def test_log_missing_countries(caplog, activity):
    """Configured countries without data are logged, not filled."""
    stock = extract_stock(activity, CC, "XXX", "s")
    with caplog.at_level("WARNING"):
        missing = log_missing_countries(stock, ["XXX", "GBR"])
    assert missing == ["GBR"]
    assert "GBR" in caplog.text
