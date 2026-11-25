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


FILL_METHODS = ["observed", "interpolated", "neighbour_ratio"]
"""How a value in a filled national table was obtained (step 5, gap filling)."""


def _filled(schema: pa.DataFrameSchema) -> pa.DataFrameSchema:
    """A national schema plus a `filled` column, one value per country, year and category."""
    return pa.DataFrameSchema(
        {**schema.columns, "filled": pa.Column(str, pa.Check.isin(FILL_METHODS))},
        unique=["country_id", "year", "vehicle_category"],
        strict=True,
        ordered=True,
        coerce=False,
    )


FilledVkmSchema = _filled(NationalVkmSchema)
"""National road vehicle-km after gap filling (observed and filled values)."""

FilledStockSchema = _filled(NationalStockSchema)
"""National road vehicle stock after gap filling (observed and filled values)."""

_NULLABLE_VKM = {"nullable": True, "checks": pa.Check.ge(0)}

HeavyDutyLevelsSchema = pa.DataFrameSchema(
    {
        "country_id": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{3}$")),
        "year": pa.Column(int, pa.Check.in_range(1990, 2100)),
        **{
            name: pa.Column(float, **_NULLABLE_VKM)
            for name in ["LNAT", "ENAT", "CAB", "D", "I_f", "coverage_m", "I_f_corrected",
                         "J_o", "I_o", "alpha_E", "s_long", "L_int_foreign", "L_int_own",
                         "L_int", "L_dom", "L_long", "L_short", "total", "T"]
        },
        "unit": pa.Column(str, pa.Check.eq("vkm")),
        "source": pa.Column(str),
        "flags": pa.Column(str),
    },
    unique=["country_id", "year"],
    strict=True,
    ordered=True,
    coerce=False,
)
"""Heavy-duty national levels per country and year before gap filling (MULTIYEAR_PLAN.md §6.3-§6.8).

vkm columns (territorial): L_int (international, incl. transit), L_dom = L_long + L_short
(domestic), total = L_int + L_dom. Statistics: LNAT/ENAT (own hauliers' national laden/
empty vkm), CAB (foreign cabotage vkm), D (domestic statistic), I_f / I_f_corrected (foreign
hauliers' international tonnes), J_o (own hauliers' international tonnes), I_o (own
hauliers' international vkm), alpha_E (laden vkm share of journeys >= 50 km), s_long
(long-haul share of domestic vkm, = alpha_E), T (territorial total).
"""

HeavyDutyCalibrationSchema = pa.DataFrameSchema(
    {
        "country_id": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{3}$")),
        "reference_year": pa.Column(int),
        **{
            name: pa.Column(float, **_NULLABLE_VKM)
            for name in ["etis_int", "etis_dom", "etis_int_legs", "etis_dom_legs", "L_int0", "L_dom0", "w0", "r_f", "r_o",
                         "gamma"]
        },
        "calibration_flags": pa.Column(str),
    },
    unique=["country_id"],
    strict=True,
    ordered=True,
    coerce=False,
)
"""Heavy-duty reference-year calibration per country (MULTIYEAR_PLAN.md §6.3)."""
