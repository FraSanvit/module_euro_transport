"""Pandera schemas for tabular module data."""

import pandera.pandas as pa

VEHICLE_CATEGORIES = ["heavy_duty", "light_duty", "passenger_car", "motorcycle", "bus"]
"""Road vehicle categories covered by the module."""

PRINCIPLES = ["territorial", "nationality", "fuel_sold", "unspecified"]
"""Accounting principles a national total can follow (see WORKFLOW_SPEC §3)."""

NationalVkmSchema = pa.DataFrameSchema(
    {
        "country_id": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{3}$")),
        "year": pa.Column(int, pa.Check.in_range(1990, 2100)),
        "vehicle_category": pa.Column(str, pa.Check.isin(VEHICLE_CATEGORIES)),
        "vkm": pa.Column(float, pa.Check.ge(0)),
        "unit": pa.Column(str, pa.Check.eq("vkm")),
        "source": pa.Column(str),
        "principle": pa.Column(str, pa.Check.isin(PRINCIPLES)),
    },
    unique=["country_id", "year", "vehicle_category", "source"],
    strict=True,
    ordered=True,
    coerce=False,
)
"""Annual national road vehicle-km per vehicle category, one row per source."""

NationalStockSchema = pa.DataFrameSchema(
    {
        "country_id": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{3}$")),
        "year": pa.Column(int, pa.Check.in_range(1990, 2100)),
        "vehicle_category": pa.Column(str, pa.Check.isin(VEHICLE_CATEGORIES)),
        "stock": pa.Column(float, pa.Check.ge(0)),
        "unit": pa.Column(str, pa.Check.eq("vehicles")),
        "source": pa.Column(str),
    },
    unique=["country_id", "year", "vehicle_category", "source"],
    strict=True,
    ordered=True,
    coerce=False,
)
"""National road vehicle stock (number of vehicles) per vehicle category and source."""
