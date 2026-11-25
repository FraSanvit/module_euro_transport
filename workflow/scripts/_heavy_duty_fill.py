"""Gap filling of the heavy-duty national levels (MULTIYEAR_PLAN.md §6.9).

Three series are filled per country and year: the domestic level D (`L_dom`), the
international level `L_int` and the long-haul share `alpha_E`. Pass 1 fills each
country from its own series (a ratio interpolated inside the series, or extrapolated
at its ends with a clipped Theil-Sen trend); pass 2 fills countries without a series
from their neighbours, which are complete after pass 1. Only ratios and shares are
carried between years or countries, never levels.
"""

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
OBSERVED = "observed"
INTERPOLATED = "interpolated"
OWN_TREND = "own_trend"
OUTLIER = "outlier_replaced"
GDP_PPS = "neighbours_gdp_pps"
GDP_EUR = "neighbours_gdp_eur"
NEIGHBOUR_RATIO = "neighbour_ratio"
NEIGHBOUR_MEAN = "neighbour_mean"
FILL_LABELS = [OBSERVED, INTERPOLATED, OWN_TREND, OUTLIER, GDP_PPS, GDP_EUR, NEIGHBOUR_RATIO, NEIGHBOUR_MEAN]


def theil_sen(x, y) -> tuple[float, float]:
    """Slope (median of the slopes between all pairs) and intercept (median of y - slope x).

    With one point the slope is 0 (the value is held).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0, float(y[0])
    i, j = np.triu_indices(len(x), k=1)
    slope = float(np.median((y[j] - y[i]) / (x[j] - x[i])))
    return slope, float(np.median(y - slope * x))


def fill_series(series: pd.Series, years, max_inner: int = 3, window: int = 5, name: str = ""):
    """Fill a yearly series over `years`.

    Inner gaps: linear interpolation (gaps longer than `max_inner` are interpolated too,
    with a warning). Gaps at the start or end: Theil-Sen trend over the nearest `window`
    observed years, clipped to the range of those years. Returns `(values, labels)`
    indexed by `years`; both stay empty/missing if nothing is observed.
    """
    years = pd.Index(sorted(years), name="year")
    s = series.reindex(years).astype(float)
    labels = pd.Series(np.where(s.notna(), OBSERVED, ""), index=years, dtype=object)
    observed = s.dropna()
    if observed.empty:
        return s, labels
    out = s.copy()
    first, last = observed.index.min(), observed.index.max()
    inner = s.isna() & (years > first) & (years < last)
    if inner.any():
        out[inner] = np.interp(years[inner].to_numpy(float), observed.index.to_numpy(float), observed.to_numpy())
        labels[inner] = INTERPOLATED
        runs = (inner != inner.shift(fill_value=False)).cumsum()[inner]
        for _, run in runs.groupby(runs):
            if len(run) > max_inner:
                LOGGER.warning(
                    "%s: gap %d-%d of %d years > %d interpolated.", name, run.index[0], run.index[-1], len(run), max_inner
                )
    for edge, reference in [(years > last, observed.iloc[-window:]), (years < first, observed.iloc[:window])]:
        if not edge.any():
            continue
        slope, intercept = theil_sen(reference.index, reference.to_numpy())
        values = intercept + slope * years[edge].to_numpy(float)
        out[edge] = np.clip(values, reference.min(), reference.max())
        labels[edge] = OWN_TREND
    for year in years[(labels != OBSERVED) & (labels != "")]:
        LOGGER.info("%s %d: %.6g (%s).", name, year, out[year], labels[year])
    return out, labels


def drop_outliers(series: pd.Series, n_mad: float = 4.0, min_abs: float = 0.1, name: str = ""):
    """Set values far from the series' own median to missing.

    An outlier is more than `n_mad` robust standard deviations (1.4826 x MAD) and more
    than `min_abs` from the median. Returns `(series, outlier mask)`.
    """
    deviation = (series - series.median()).abs()
    scale = 1.4826 * deviation.median()
    bad = (deviation > n_mad * scale) & (deviation > min_abs)
    for year, value in series[bad].items():
        LOGGER.warning("%s %d: %.4g is an outlier (own median %.4g); replaced.", name, year, value, series.median())
    return series.where(~bad), bad


def neighbour_mean(table: pd.DataFrame, neighbours: list[str]) -> pd.Series:
    """Mean over the neighbours (columns of `table`) per year, ignoring missing values."""
    present = [n for n in neighbours if n in table.columns]
    return table[present].mean(axis=1, skipna=True) if present else pd.Series(np.nan, index=table.index)


def fill_levels(
    levels: pd.DataFrame,
    gdp: pd.DataFrame,
    years: list[int],
    neighbours: dict,
    max_inner: int = 3,
    window: int = 5,
    outlier_mad: float = 4.0,
    outlier_min_abs: float = 0.1,
) -> pd.DataFrame:
    """Fill `L_dom`, `L_int` and `alpha_E` for every country and year (§6.9).

    `levels` from `heavy_duty_national_levels` (ISO3 `country_id`); `gdp` has
    `country_id, year, gdp_pps, gdp_eur`. Returns one row per country and year with
    the filled series, `L_long, L_short, total`, the diagnostic columns and the labels
    `filled_L_dom, filled_L_int, filled_alpha_E`.
    """
    years = sorted(years)
    countries = sorted(levels["country_id"].unique())
    lv = levels.set_index(["country_id", "year"])
    gdp_filled = {}
    for column in ["gdp_pps", "gdp_eur"]:
        wide = gdp.pivot_table(index="year", columns="country_id", values=column)
        gdp_filled[column] = pd.DataFrame(
            {c: fill_series(wide[c].dropna(), years, max_inner, window, f"{c} {column}")[0]
             for c in wide.columns if c in countries}  # fmt: skip
        ).reindex(index=years, columns=countries)
    frames = {}
    for c in countries:
        f = lv.loc[c].reindex(years) if c in lv.index.get_level_values(0) else pd.DataFrame(index=years)
        frames[c] = f.copy()
        for column in ["filled_L_dom", "filled_L_int", "filled_alpha_E"]:
            frames[c][column] = ""

    def column(name):
        return pd.DataFrame({c: frames[c][name] for c in countries}, index=years)

    # Pass 1a: long-haul share and domestic level from the country's own series.
    for c in countries:
        f = frames[c]
        alpha = f["alpha_E"].dropna()
        if not alpha.empty:
            clean, bad = drop_outliers(alpha, outlier_mad, outlier_min_abs, f"{c} alpha_E")
            values, labels = fill_series(clean.dropna(), years, max_inner, window, f"{c} alpha_E")
            labels[bad[bad].index] = OUTLIER
            f["alpha_E"], f["filled_alpha_E"] = values, labels
        override = f["T"].notna()
        if override.any():
            # Territorial total: fill L_int / T, then D = T - L_int (§6.8).
            ratio, labels = fill_series((f["L_int"] / f["T"]).dropna(), years, max_inner, window, f"{c} L_int/T")
            missing = f["L_int"].isna() & override
            f.loc[missing, "L_int"] = (ratio * f["T"])[missing]
            f.loc[missing, "L_dom"] = (f["T"] - f["L_int"])[missing]
            f["filled_L_int"] = labels.where(missing, np.where(f["L_int"].notna(), OBSERVED, ""))
            f["filled_L_dom"] = f["filled_L_int"]
        elif f["L_dom"].notna().any():
            gdp_c = gdp_filled["gdp_pps"][c]
            ratio, labels = fill_series((f["L_dom"] / gdp_c).dropna(), years, max_inner, window, f"{c} D/GDP_PPS")
            f["L_dom"] = f["L_dom"].fillna(ratio * gdp_c)
            f["filled_L_dom"] = labels

    # Pass 2a: domestic level of countries without a series, from neighbours' D per GDP.
    for c in countries:
        f = frames[c]
        if f["L_dom"].notna().all():
            continue
        names = neighbours.get(c)
        if not names:
            LOGGER.warning("%s: domestic level missing and no neighbours; left missing.", c)
            continue
        pps = gdp_filled["gdp_pps"][c]
        key, label = ("gdp_pps", GDP_PPS) if pps.notna().any() else ("gdp_eur", GDP_EUR)
        if key == "gdp_eur":
            LOGGER.warning("%s: no GDP in PPS; domestic level from GDP in euro.", c)
        intensity = neighbour_mean(column("L_dom") / gdp_filled[key], names)
        missing = f["L_dom"].isna()
        f.loc[missing, "L_dom"] = (intensity * gdp_filled[key][c])[missing]
        f.loc[missing, "filled_L_dom"] = label
        for year in f.index[missing]:
            LOGGER.info("%s %d: D %.4g vkm (%s %s).", c, year, f.loc[year, "L_dom"], label, names)

    # Pass 1b: international level from the country's own series (D is complete now).
    for c in countries:
        f = frames[c]
        missing = f["L_int"].isna()
        if not missing.any() or not f["L_int"].notna().any():
            if not missing.any():
                f["filled_L_int"] = f["filled_L_int"].replace("", OBSERVED)
            continue
        own = f["L_int_own"]
        if own[missing].notna().all():
            ratio, labels = fill_series(
                (f["L_int_foreign"] / f["L_dom"]).dropna(), years, max_inner, window, f"{c} L_int_foreign/D"
            )
            f.loc[missing, "L_int_foreign"] = (ratio * f["L_dom"])[missing]
            f.loc[missing, "L_int"] = (f["L_int_foreign"] + own)[missing]
        else:
            ratio, labels = fill_series((f["L_int"] / f["L_dom"]).dropna(), years, max_inner, window, f"{c} L_int/D")
            f.loc[missing, "L_int"] = (ratio * f["L_dom"])[missing]
        f["filled_L_int"] = labels.where(missing, OBSERVED)

    # Pass 2b: international level and long-haul share of countries without a series.
    for c in countries:
        f = frames[c]
        names = neighbours.get(c) or []
        if f["L_int"].isna().any() and names:
            share = neighbour_mean(column("L_int") / column("L_dom"), names)
            missing = f["L_int"].isna()
            f.loc[missing, "L_int"] = (share * f["L_dom"])[missing]
            f.loc[missing, "filled_L_int"] = NEIGHBOUR_RATIO
            for year in f.index[missing]:
                LOGGER.info("%s %d: L_int %.4g vkm (%s %s).", c, year, f.loc[year, "L_int"], NEIGHBOUR_RATIO, names)
        if f["alpha_E"].isna().any() and names:
            mean = neighbour_mean(column("alpha_E"), names)
            missing = f["alpha_E"].isna()
            f.loc[missing, "alpha_E"] = mean[missing]
            f.loc[missing, "filled_alpha_E"] = NEIGHBOUR_MEAN
            LOGGER.info("%s: alpha_E from neighbours %s in %d years.", c, names, int(missing.sum()))

    rows = []
    for c in countries:
        f = frames[c].copy()
        f["L_long"] = f["alpha_E"] * f["L_dom"]
        f["L_short"] = f["L_dom"] - f["L_long"]
        f["total"] = f["L_dom"] + f["L_int"]
        rows.append(f.assign(country_id=c).reset_index())
    out = pd.concat(rows, ignore_index=True)
    for name in ["filled_L_dom", "filled_L_int", "filled_alpha_E"]:
        out[name] = out[name].replace("", OBSERVED).where(out[name.removeprefix("filled_")].notna(), "")
    return out
