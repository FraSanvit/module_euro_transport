"""Heavy-duty (HGV > 3.5 t) vehicle-km: the national level chain of MULTIYEAR_PLAN.md.

Pure functions on tidy tables; the rule scripts read files and call them. Eurostat
2-letter codes (`geo`, EL for Greece, UK for the United Kingdom) are used inside;
results are mapped to ISO 3166-1 alpha-3 at the end.

Notation (MULTIYEAR_PLAN.md §5): LNAT/ENAT laden/empty vkm of national transport by
trucks registered in a country, LINTL/EINTL the same for international transport,
CAB cabotage vkm by foreign trucks, D the domestic level, I^f the foreign hauliers'
international activity index (tonnes), J^o the own hauliers' international tonnes
loaded or unloaded in the country, I^o the own hauliers' international vkm.
"""

import hashlib
import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
MIO = 1e6
THS = 1e3
EU_AGGREGATE = "EU27_2020"
ALL_YEARS = "all"


def sha256(path, chunk=1 << 24) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path, expected: str) -> None:
    """Raise if the file's SHA-256 differs from `expected`."""
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 {actual} differs from the expected {expected}.")
    LOGGER.info("%s: SHA-256 verified.", path)


def is_country(codes: pd.Series) -> pd.Series:
    """True for 2-letter country codes (drops EU15, EU27_2020, EFTA, ...)."""
    return codes.str.len() == 2


def read_eurostat(path, keep: list[str], filters: dict | None = None) -> pd.DataFrame:
    """Read an SDMX-CSV file; columns `*keep, year, value` (empty values dropped)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    for update in sorted(raw.get("LAST UPDATE", pd.Series(dtype=str)).unique()):
        LOGGER.info("%s: Eurostat last update %s", path, update)
    for dimension, value in (filters or {}).items():
        raw = raw[raw[dimension] == value]
    raw = raw[raw["OBS_VALUE"].str.strip() != ""]
    table = raw[keep].copy()
    table["year"] = raw["TIME_PERIOD"].astype(int)
    table["value"] = raw["OBS_VALUE"].astype(float)
    if table.duplicated([*keep, "year"]).any():
        raise ValueError(f"{path}: duplicated observations for {keep} and {filters}.")
    return table.reset_index(drop=True)


# ----------------------------------------------------------------------------- survey


def survey_activity(tott: pd.DataFrame) -> pd.DataFrame:
    """Own hauliers' activity per `geo, year` from road_go_ta_tott (tra_type TOTAL).

    `tott` has columns `geo, tra_oper, unit, year, value` (million vkm / tkm).
    Returns LNAT, ENAT, LINTL, EINTL, CAB_OUT_VKM, CAB_OUT_TKM in vkm / tkm, where
    CAB_OUT is cabotage done abroad by the country's hauliers (LINTL_CAB).
    """
    vkm = tott[tott["unit"] == "MIO_VKM"].pivot_table(
        index=["geo", "year"], columns="tra_oper", values="value"
    )
    tkm = tott[tott["unit"] == "MIO_TKM"].pivot_table(
        index=["geo", "year"], columns="tra_oper", values="value"
    )
    out = pd.DataFrame(index=vkm.index.union(tkm.index))
    for oper in ["LNAT", "ENAT", "LINTL", "EINTL"]:
        out[oper] = vkm.get(oper, pd.Series(dtype=float)).reindex(out.index) * MIO
    out["CAB_OUT_VKM"] = vkm.get("LINTL_CAB", pd.Series(dtype=float)).reindex(out.index) * MIO
    out["CAB_OUT_TKM"] = tkm.get("LINTL_CAB", pd.Series(dtype=float)).reindex(out.index) * MIO
    return out


EMPTY_PAIRS = [("LNAT", "ENAT", "NAT"), ("LINTL", "EINTL", "INTL")]


def laden_trip_length(vm: pd.DataFrame) -> pd.DataFrame:
    """Average laden journey length (km) per `geo, year` from road_go_ta_vm.

    `vm` has columns `geo, tra_type, tra_cov, loadstat, unit, year, value` (MIO_VKM and
    THS_JRNY). Returns columns `trip_NAT`, `trip_INTL` (all trucks, hire + own account).
    """
    rows = vm[(vm["tra_type"] == "TOTAL") & (vm["loadstat"] == "LOADED")]
    table = rows.pivot_table(index=["geo", "year"], columns=["tra_cov", "unit"], values="value")
    out = pd.DataFrame(index=table.index)
    for cov in ["NAT", "INTL"]:
        vkm = table.get((cov, "MIO_VKM"), pd.Series(np.nan, index=table.index))
        trips = table.get((cov, "THS_JRNY"), pd.Series(np.nan, index=table.index))
        out[f"trip_{cov}"] = (vkm * MIO / (trips * THS)).where(trips > 0)
    return out


def estimate_missing_empty(survey: pd.DataFrame, trips: pd.DataFrame, bandwidth: float = 0.2) -> pd.DataFrame:
    """Estimate unreported empty km (laden reported, empty not); national and international separately.

    1. The country reports empty km in other years (BE, LU, LV, NL in early years): its
       own empty/laden ratio from the nearest reported year (later year on ties).
    2. It never does (IT, RO): a weighted mean of the ratios of the countries reporting
       both in the same year, with Gaussian weights on the difference in log average
       laden journey length (`laden_trip_length`), standard deviation `bandwidth`. Empty
       running depends on trip length (short distribution trips return empty more
       often), so trip length is a better guide than geography; smooth weights avoid
       jumps when the closest countries change between years.
    Rows without either stay missing (see `zero_missing_empty`). Columns
    `empty_est_nat`, `empty_est_intl` mark the estimates.
    """
    survey = survey.copy()
    trips = trips.reindex(survey.index)
    geo = survey.index.get_level_values("geo").to_series(index=survey.index)
    country = is_country(geo)
    for laden, empty, cov in EMPTY_PAIRS:
        flag = f"empty_est_{cov.lower()}"
        survey[flag] = False
        trip = trips[f"trip_{cov}"]
        own = country & survey[empty].notna() & (survey[laden] > 0)
        reported = own & trip.notna()
        missing = country & survey[laden].notna() & survey[empty].isna()
        ratio = survey[empty] / survey[laden]
        for idx in survey.index[missing]:
            year = idx[1]
            own_years = survey.index[own & (geo == idx[0])].get_level_values("year")
            if len(own_years):
                nearest = min(own_years, key=lambda y: (abs(y - year), -y))
                estimate = ratio[(idx[0], nearest)]
                survey.loc[idx, empty] = survey.loc[idx, laden] * estimate
                survey.loc[idx, flag] = True
                LOGGER.info("%s %d %s: own empty/laden %.3f of %d.", idx[0], year, empty, estimate, nearest)
                continue
            if np.isnan(trip[idx]):
                continue
            ref = reported & (survey.index.get_level_values("year") == year)
            if not ref.any():
                continue
            distance = np.log(trip[ref]) - np.log(trip[idx])
            weight = np.exp(-0.5 * (distance / bandwidth) ** 2)
            estimate = (weight * ratio[ref]).sum() / weight.sum()
            survey.loc[idx, empty] = survey.loc[idx, laden] * estimate
            survey.loc[idx, flag] = True
            top = (weight / weight.sum()).sort_values(ascending=False).index[:5]
            LOGGER.info(
                "%s %d %s: laden trip %.0f km; empty/laden %.3f; largest weights %s.",
                idx[0], year, empty, trip[idx], estimate,
                ", ".join(
                    f"{g} ({trip[(g, y)]:.0f} km, ratio {ratio[(g, y)]:.2f}, w {weight[(g, y)] / weight.sum():.2f})"
                    for g, y in top
                ),
            )  # fmt: skip
        hit = survey.index[survey[flag]].to_frame(index=False)
        for g, years in hit.groupby("geo")["year"]:
            LOGGER.warning(
                "%s: %s not reported in %d-%d; estimated (own ratio of the nearest year, else countries with a similar laden trip length).",
                g, empty, years.min(), years.max(),
            )  # fmt: skip
    return survey


def zero_missing_empty(survey: pd.DataFrame) -> pd.DataFrame:
    """Set empty km to 0 where laden km are reported but empty km are not.

    Fallback after `estimate_missing_empty`. Gives a lower bound for the domestic
    level (and for cabotage hosted there, whose empty share is the host's). Columns `empty_zero_nat` (ENAT) and `empty_zero_intl`
    (EINTL) mark the rows; logged per country.
    """
    survey = survey.copy()
    for laden, empty, column in [("LNAT", "ENAT", "empty_zero_nat"), ("LINTL", "EINTL", "empty_zero_intl")]:
        missing = survey[laden].notna() & survey[empty].isna()
        survey.loc[missing, empty] = 0.0
        survey[column] = missing
        hit = survey.index[missing].to_frame(index=False)
        hit = hit[is_country(hit["geo"])]
        for geo, years in hit.groupby("geo")["year"]:
            LOGGER.warning(
                "%s: no %s reported in %d-%d; set to 0 (lower bound).",
                geo, empty, years.min(), years.max(),
            )  # fmt: skip
    return survey


def short_trip_share(dc: pd.DataFrame, survey: pd.DataFrame) -> pd.Series:
    """alpha_E = 1 - laden vkm of journeys < 50 km / LNAT (§6.7), per `geo, year`.

    `dc` has columns `geo, distance, year, value` (million vkm, tra_type TOTAL).
    """
    lt50 = dc[dc["distance"] == "KM_LT50"].set_index(["geo", "year"])["value"] * MIO
    lnat = survey["LNAT"].reindex(lt50.index)
    alpha = (1 - lt50 / lnat).where(lnat > 0)
    bad = alpha[(alpha < 0) | (alpha > 1)]
    for (geo, year), value in bad.items():
        LOGGER.warning(
            "%s %d: journeys < 50 km exceed national laden vkm (alpha_E %.3f); "
            "alpha_E left missing (alpha_max used).", geo, year, value,
        )  # fmt: skip
    return alpha.drop(bad.index).dropna().rename("alpha_E")


# ----------------------------------------------------------------------------- cabotage


def cabotage_load_factor(survey: pd.DataFrame) -> pd.DataFrame:
    """Laden load factor of each haulier country's cabotage [t per laden vkm] (§6.5).

    Own value where finite and positive, else the EU27 value of the same year.
    Returns `geo, year, load_factor, lf_source`.
    """
    lf = (survey["CAB_OUT_TKM"] / survey["CAB_OUT_VKM"]).replace([np.inf, -np.inf], np.nan)
    lf = lf.where(lf > 0)
    eu = lf.xs(EU_AGGREGATE, level="geo") if EU_AGGREGATE in lf.index.get_level_values(0) else pd.Series(dtype=float)
    frame = lf.rename("own").reset_index()
    frame["eu"] = frame["year"].map(eu)
    frame["load_factor"] = frame["own"].fillna(frame["eu"])
    frame["lf_source"] = np.where(frame["own"].notna(), "own", "EU27")
    return frame[["geo", "year", "load_factor", "lf_source"]]


def cabotage_vkm(
    hac: pd.DataFrame,
    survey: pd.DataFrame,
    empty_share="host_national",
    hosts_zero=(),
) -> pd.DataFrame:
    """Foreign trucks' cabotage vkm per host country and year (§6.5).

    `hac` (road_go_ca_hac, unit THS_TKM) has columns `geo` (haulier), `c_cabot` (host),
    `year`, `value` (thousand tkm). CAB = sum over hauliers h != host of
    tkm / LF_h x (1 + e_host); e_host = ENAT / LNAT of the host (`host_national`) or a
    fixed share. Hosts in `hosts_zero` get 0. Returns `geo, year, CAB, CAB_laden`.
    """
    flows = hac[is_country(hac["geo"]) & is_country(hac["c_cabot"])]
    flows = flows[flows["geo"] != flows["c_cabot"]]
    lf = cabotage_load_factor(survey)
    flows = flows.merge(lf, on=["geo", "year"], how="left")
    no_lf = flows[flows["load_factor"].isna()]
    for (haulier, year), _ in no_lf.groupby(["geo", "year"]):
        LOGGER.warning("Cabotage: no load factor for hauliers %s in %d (their cabotage left out).", haulier, year)
    flows["laden_vkm"] = flows["value"] * THS / flows["load_factor"]
    laden = flows.groupby(["c_cabot", "year"])["laden_vkm"].sum(min_count=1)
    laden.index = laden.index.set_names(["geo", "year"])
    if empty_share == "host_national":
        empty = (survey["ENAT"] / survey["LNAT"]).reindex(laden.index)
    else:
        empty = pd.Series(float(empty_share), index=laden.index)
    out = pd.DataFrame({"CAB_laden": laden, "CAB": laden * (1 + empty)}).reset_index()
    out.loc[out["geo"].isin(hosts_zero), ["CAB", "CAB_laden"]] = 0.0
    return out


# ----------------------------------------------------------------------------- international index


def international_tonnes(ia: pd.DataFrame) -> pd.DataFrame:
    """Tonnes loaded or unloaded in each country by international transport (§6.3b, §6.6b).

    `ia` (road_go_ia_rc, unit THS_T) has columns `geo` (reporting haulier), `c_load`,
    `c_unload`, `year`, `value` (thousand t). Returns per `country, haulier, year` the
    tonnes of international transports with one end in `country` and the other abroad.
    """
    rows = ia[is_country(ia["geo"]) & is_country(ia["c_load"]) & is_country(ia["c_unload"])]
    rows = rows[rows["c_load"] != rows["c_unload"]]
    loaded = rows.rename(columns={"c_load": "country"})[["country", "geo", "year", "value"]]
    unloaded = rows.rename(columns={"c_unload": "country"})[["country", "geo", "year", "value"]]
    both = pd.concat([loaded, unloaded], ignore_index=True)
    out = both.groupby(["country", "geo", "year"], as_index=False)["value"].sum()
    out["tonnes"] = out.pop("value") * THS
    return out.rename(columns={"geo": "haulier"})


def transit_tonnes(tra: pd.DataFrame) -> pd.DataFrame:
    """Transit tonnes through each country per reporting haulier (road_go_ta_tra, THS_T)."""
    rows = tra[is_country(tra["geo"]) & is_country(tra["c_trans"])]
    out = rows.rename(columns={"c_trans": "country", "geo": "haulier"})
    out = out.groupby(["country", "haulier", "year"], as_index=False)["value"].sum()
    out["tonnes"] = out.pop("value") * THS
    return out


def international_index(
    ia: pd.DataFrame, tra: pd.DataFrame, reference_year: int, correct_coverage=True,
    max_coverage_gap=0.5,
) -> pd.DataFrame:
    """Foreign and own international activity per country and year.

    I^f = tonnes loaded or unloaded in the country by foreign hauliers + transit tonnes
    of foreign hauliers through it; J^o = international tonnes loaded or unloaded in
    the country by its own hauliers. Coverage correction (§6.6): I^f / (1 - m), m = share
    of I^f at the reference year from hauliers that do not report in the year.
    Returns `geo, year, I_f, J_o, coverage_m, I_f_corrected`.
    """
    lu = international_tonnes(ia).assign(kind="load_unload")
    tr = transit_tonnes(tra).assign(kind="transit")
    contrib = pd.concat([lu, tr], ignore_index=True)
    reporters = contrib.groupby("year")["haulier"].agg(set)
    own = lu[lu["country"] == lu["haulier"]].groupby(["country", "year"])["tonnes"].sum()
    foreign = contrib[contrib["country"] != contrib["haulier"]]
    i_f = foreign.groupby(["country", "year"])["tonnes"].sum()
    out = pd.DataFrame({"I_f": i_f, "J_o": own}).rename_axis(["geo", "year"])
    out["coverage_m"] = 0.0
    if correct_coverage and reference_year in reporters.index:
        base = foreign[foreign["year"] == reference_year]
        by_haulier = base.groupby(["country", "haulier"])["tonnes"].sum()
        total = by_haulier.groupby(level="country").sum()
        for year, present in reporters.items():
            missing = by_haulier[~by_haulier.index.get_level_values("haulier").isin(present)]
            if missing.empty:
                continue
            share = (missing.groupby(level="country").sum() / total).dropna()
            absent = sorted(set(missing.index.get_level_values("haulier")))
            LOGGER.info("International index %d: reporters absent vs %d: %s", year, reference_year, absent)
            idx = [(c, year) for c in share.index if (c, year) in out.index]
            out.loc[idx, "coverage_m"] = share.reindex([c for c, _ in idx]).to_numpy()
    out["I_f_corrected"] = out["I_f"] / (1 - out["coverage_m"])
    too_low = out["coverage_m"] > max_coverage_gap
    for geo, year in out.index[too_low]:
        LOGGER.warning(
            "%s %d: %.0f%% of the foreign-hauliers index at the reference year comes from "
            "hauliers not reporting; index left missing.", geo, year, 100 * out.loc[(geo, year), "coverage_m"],
        )  # fmt: skip
    out.loc[too_low, "I_f_corrected"] = np.nan
    return out.reset_index()


# ----------------------------------------------------------------------------- statistics table


def statistics(
    tott, dc, hac, ia, tra, reference_year, empty_share="host_national", hosts_zero=(),
    correct_coverage=True, max_coverage_gap=0.5, vm=None, empty_bandwidth=0.2,
) -> pd.DataFrame:
    """All annual statistics per Eurostat country and year (§6.4-§6.7 inputs).

    `vm` (road_go_ta_vm) gives laden trip lengths for estimating unreported empty km;
    without it, unreported empty km are 0 (lower bound).
    """
    survey = survey_activity(tott)
    if vm is not None:
        survey = estimate_missing_empty(survey, laden_trip_length(vm), empty_bandwidth)
    survey = zero_missing_empty(survey)
    cab = cabotage_vkm(hac, survey, empty_share, hosts_zero).set_index(["geo", "year"])
    alpha = short_trip_share(dc, survey)
    index = international_index(
        ia, tra, reference_year, correct_coverage, max_coverage_gap
    ).set_index(["geo", "year"])
    table = survey.join(cab, how="outer").join(alpha, how="outer").join(index, how="outer")
    table = table[is_country(table.index.get_level_values("geo").to_series()).to_numpy()]
    table["I_o"] = table["LINTL"] + table["EINTL"]
    return table.reset_index()


def domestic_level(stats: pd.DataFrame, include_cabotage=True, hosts_zero=()) -> pd.DataFrame:
    """D = LNAT + ENAT + CAB (§6.4); CAB is 0 for hosts without cabotage data or in `hosts_zero`.

    D stays missing where LNAT or ENAT is missing. Unreported empty km (IT, RO) are
    already 0 here (`zero_missing_empty`), so D is a lower bound there.
    """
    stats = stats.copy()
    cab = stats["CAB"].fillna(0.0) if include_cabotage else 0.0
    if include_cabotage:
        cab = cab.where(~stats["geo"].isin(hosts_zero), 0.0)
    stats["CAB"] = cab
    stats["D"] = stats["LNAT"] + stats["ENAT"] + stats["CAB"]
    return stats


def apply_exclusions(stats: pd.DataFrame, exclude: dict, codes: dict) -> pd.DataFrame:
    """Set distrusted survey values to missing (settings `exclude`), logged.

    `exclude` maps ISO3 -> {"domestic" | "own_international": years or "all"}.
    "domestic" clears LNAT, ENAT, alpha_E; "own_international" clears LINTL, EINTL,
    I_o and J_o.
    """
    columns = {
        "domestic": ["LNAT", "ENAT", "alpha_E"],
        "own_international": ["LINTL", "EINTL", "I_o", "J_o"],
    }
    stats = stats.copy()
    for iso3, parts in (exclude or {}).items():
        if iso3 not in codes:
            continue
        geo = codes[iso3]
        for part, years in parts.items():
            hit = stats["geo"] == geo
            if years != ALL_YEARS:
                hit &= stats["year"].isin(years)
            if hit.any():
                LOGGER.warning(
                    "%s %s: survey values excluded as distrusted, years %s (left missing).",
                    iso3, part, sorted(stats.loc[hit, "year"]),
                )  # fmt: skip
                stats.loc[hit, columns[part]] = np.nan
    return stats


# ----------------------------------------------------------------------------- levels


def alpha_cap(alpha_e: pd.Series, f_net: float, alpha_max: float) -> pd.Series:
    """alpha_cap = min(f_net * alpha_E, alpha_max); alpha_max where alpha_E is missing."""
    return (f_net * alpha_e).clip(upper=alpha_max).fillna(alpha_max)


def override_split(total, l_int, floor):
    """Fit components into a territorial total (§6.8).

    L_dom = min(max(T - L_int, floor), T), L_int = T - L_dom, with floor = LNAT + ENAT
    (0 where missing). Works element-wise on scalars or arrays.
    """
    floor = np.where(np.isnan(floor), 0.0, floor)
    l_dom = np.minimum(np.maximum(total - l_int, floor), total)
    return l_dom, total - l_dom


def calibrate(
    stats: pd.DataFrame,
    network: pd.DataFrame,
    totals: pd.DataFrame,
    toll: pd.DataFrame | None,
    reference_year: int,
    k: float,
    alpha_max: float,
    f_net: float,
    islands=(),
    toll_countries=(),
    toll_cabotage_share=1.0,
    countries=None,
) -> pd.DataFrame:
    """Reference-year calibration per country (§6.3). All inputs keyed by `geo`.

    `stats` from `statistics` (incl. `D`), `network` has `geo, etis_dom, etis_int`
    (vkm, reference flows), `totals` has `geo, year, T`, `toll` has `geo, year,
    toll_foreign` (vkm). `countries` lists the `geo` codes to calibrate (default: all
    in the inputs); countries without inputs get a row of missing values. Returns per
    `geo`: L_int0, L_dom0, w0, r_f, r_o, s, gamma, alpha_cap0 and `calibration_flags`.
    """
    s0 = stats[stats["year"] == reference_year].set_index("geo")
    net = network.set_index("geo")
    t0 = totals[totals["year"] == reference_year].set_index("geo")["T"]
    rows = []
    if countries is None:
        countries = set(s0.index) | set(net.index) | set(t0.index)
    for geo in sorted(countries):
        st = s0.loc[geo] if geo in s0.index else pd.Series(dtype=float)
        flags = []
        if geo not in net.index and geo not in islands:
            flags.append("no_network")
        if bool(st.get("empty_zero_intl", False)) and geo not in islands:
            flags.append("own_int_empty_zero_t0")  # r_o then scales a laden-only index
        cap0 = alpha_cap(pd.Series([st.get("alpha_E", np.nan)]), f_net, alpha_max).iloc[0]
        etis_int = net["etis_int"].get(geo, np.nan)
        etis_dom = net["etis_dom"].get(geo, np.nan)
        l_int0 = 0.0 if geo in islands else k * etis_int
        floor = st.get("LNAT", np.nan) + st.get("ENAT", np.nan)
        d0 = st.get("D", np.nan)
        if geo in t0.index and not np.isnan(l_int0):
            d0, l_int_new = override_split(t0[geo], l_int0, np.array(floor))
            d0, l_int_new = float(d0), float(l_int_new)
            if abs(l_int_new - l_int0) > 1e-6 * t0[geo]:
                flags.append("int_adjusted_t0")
            l_int0 = l_int_new
        i_f0, j_o0 = st.get("I_f", np.nan), st.get("J_o", np.nan)
        if geo in islands:
            w0 = np.nan
        elif np.isnan(j_o0):
            w0 = 1.0 if i_f0 > 0 else np.nan
            flags.append("own_hauliers_not_reporting")
        else:
            w0 = i_f0 / (i_f0 + j_o0) if (i_f0 + j_o0) > 0 else np.nan
        l_f0 = w0 * l_int0 if geo not in islands else 0.0
        l_o0 = (1 - w0) * l_int0 if geo not in islands else 0.0
        r_f = l_f0 / i_f0 if i_f0 > 0 else np.nan
        i_o0 = st.get("I_o", np.nan)
        r_o = 0.0 if l_o0 == 0 else (l_o0 / i_o0 if i_o0 > 0 else np.nan)
        s = min(k * etis_dom / d0, cap0) if d0 > 0 else np.nan
        if d0 > 0 and k * etis_dom / d0 > cap0:
            flags.append("long_capped_t0")
        gamma = np.nan
        if geo in toll_countries and toll is not None:
            tf = toll.set_index(["geo", "year"])["toll_foreign"].get((geo, reference_year), np.nan)
            cab0 = st.get("CAB", 0.0)
            gamma = l_f0 / (tf - toll_cabotage_share * cab0)
        rows.append(
            {
                "geo": geo, "etis_int": etis_int, "etis_dom": etis_dom, "L_int0": l_int0,
                "L_dom0": d0, "w0": w0, "r_f": r_f, "r_o": r_o, "s": s, "gamma": gamma,
                "alpha_cap0": cap0, "calibration_flags": ";".join(flags),
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def annual_levels(
    stats: pd.DataFrame,
    calibration: pd.DataFrame,
    totals: pd.DataFrame,
    toll: pd.DataFrame | None,
    years: list[int],
    alpha_max: float,
    f_net: float,
    islands=(),
    toll_countries=(),
    toll_cabotage_share=1.0,
    jump=0.25,
) -> pd.DataFrame:
    """National levels per country and year (§6.4-§6.8), no gap filling.

    Returns per `geo, year`: the statistics used, L_int_foreign, L_int_own, L_int,
    L_long, L_short, total, T, source and `flags`. Values stay missing where an input
    is missing (filled later in the common gap-filling step).
    """
    cal = calibration.set_index("geo")
    st = stats[stats["year"].isin(years)].set_index(["geo", "year"])
    tot = totals.set_index(["geo", "year"])["T"]
    tl = toll.set_index(["geo", "year"])["toll_foreign"] if toll is not None else pd.Series(dtype=float)
    keys = pd.MultiIndex.from_product([sorted(cal.index), years], names=["geo", "year"])
    out = st.reindex(keys)
    out["T"] = tot.reindex(keys)
    out["alpha_cap"] = alpha_cap(out["alpha_E"], f_net, alpha_max)
    geo = out.index.get_level_values("geo")
    c = cal.reindex(geo)
    flags = [[] for _ in range(len(out))]
    lnat, enat = out["LNAT"].to_numpy(), out["ENAT"].to_numpy()
    def marker(column):
        if column not in out:
            return np.zeros(len(out), dtype=bool)
        return out[column].fillna(False).astype(bool).to_numpy()

    zero_nat, zero_intl = marker("empty_zero_nat"), marker("empty_zero_intl")
    est_nat, est_intl = marker("empty_est_nat"), marker("empty_est_intl")
    cal_flags = c["calibration_flags"].astype(str).to_numpy()
    for i in range(len(out)):
        if est_nat[i] or (est_intl[i] and geo[i] not in islands):
            flags[i].append("empty_km_estimated")
        if np.isnan(lnat[i]):
            flags[i].append("no_domestic_statistic")
        elif np.isnan(enat[i]):
            flags[i].append("empty_km_missing")
        if zero_nat[i] and not np.isnan(lnat[i]):
            flags[i].append("empty_km_zero_lower_bound")
        # Own international: a lower bound only where the reference year had empty km
        # (otherwise r_o was calibrated on the same laden-only index).
        if zero_intl[i] and "own_int_empty_zero_t0" not in cal_flags[i] and geo[i] not in islands:
            flags[i].append("own_int_empty_km_zero_lower_bound")
        if "no_network" in str(c["calibration_flags"].iloc[i]):
            flags[i].append("no_network")
        if not np.isnan(out["I_f"].iloc[i]) and np.isnan(out["I_f_corrected"].iloc[i]):
            flags[i].append("index_coverage_too_low")

    l_f = c["r_f"].to_numpy() * out["I_f_corrected"].to_numpy()
    use_toll = np.array([g in toll_countries for g in geo]) & ~np.isnan(tl.reindex(out.index).to_numpy())
    toll_lf = c["gamma"].to_numpy() * (
        tl.reindex(out.index).to_numpy() - toll_cabotage_share * out["CAB"].fillna(0).to_numpy()
    )
    l_f = np.where(use_toll, toll_lf, l_f)
    l_o = np.where(c["r_o"].to_numpy() == 0, 0.0, c["r_o"].to_numpy() * out["I_o"].to_numpy())
    island = np.array([g in islands for g in geo])
    l_f = np.where(island, 0.0, l_f)
    l_o = np.where(island, 0.0, l_o)
    l_int = l_f + l_o
    d = out["D"].to_numpy()
    s = c["s"].to_numpy()
    cap = out["alpha_cap"].to_numpy()
    l_long = np.minimum(s * d, cap * d)
    total = l_int + d
    source = np.where(use_toll, "survey + toll", "survey + index")

    t = out["T"].to_numpy()
    ov = ~np.isnan(t)
    floor = (out["LNAT"] + out["ENAT"]).to_numpy()
    l_dom_ov, l_int_ov = override_split(t, l_int, floor)
    l_dom_ov = np.where(np.isnan(l_int), np.nan, l_dom_ov)
    l_int_ov = np.where(np.isnan(l_int), np.nan, l_int_ov)
    for i in np.flatnonzero(ov):
        if np.isnan(l_int[i]):
            flags[i].append("split_missing")
        elif abs(l_int_ov[i] - l_int[i]) > 1e-6 * t[i]:
            flags[i].append("int_adjusted")
    l_long_ov = np.minimum(s * l_dom_ov, cap * l_dom_ov)
    d = np.where(ov, l_dom_ov, d)
    l_int = np.where(ov, l_int_ov, l_int)
    l_long = np.where(ov, l_long_ov, l_long)
    total = np.where(ov, t, total)
    source = np.where(ov, "territorial total", source)
    for i in np.flatnonzero(s * d > cap * d):
        flags[i].append("long_capped")
    for i in np.flatnonzero(np.isnan(out["alpha_E"].to_numpy())):
        flags[i].append("alpha_max_used")

    out["L_int_foreign"] = np.where(ov, np.nan, l_f)
    out["L_int_own"] = np.where(ov, np.nan, l_o)
    out["L_int"] = l_int
    out["L_dom"] = d
    out["L_long"] = l_long
    out["L_short"] = d - l_long
    out["total"] = total
    out["level_source"] = source
    out["flags"] = [";".join(f) for f in flags]
    out = out.reset_index()
    for column in ["D", "I_f", "CAB"]:
        change = out.groupby("geo")[column].pct_change(fill_method=None).abs()
        for i in np.flatnonzero(change > jump):
            out.loc[i, "flags"] = ";".join(filter(None, [out.loc[i, "flags"], f"jump_{column}"]))
    return out
