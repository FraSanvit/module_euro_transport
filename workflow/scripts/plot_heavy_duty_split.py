"""Diagnostic plot: heavy-duty vehicle-km per country and year, split by component.

One panel per country (own y scale), stacked bars per year from the filled national
levels (MULTIYEAR_PLAN.md §6.9): domestic short-haul, domestic long-haul,
international by own hauliers and by foreign hauliers. International vkm without a
haulier split (territorial totals, values filled from ratios or neighbours) is one
hatched segment. Segments that are filled rather than observed are drawn lighter.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
FILLED_ALPHA = 0.4

# Categorical slots 1-4 in fixed order (validated adjacent-pair palette).
SEGMENTS = [
    ("L_short", "Domestic, short-haul (<50 km)", "#2a78d6", None, "domestic"),
    ("L_long", "Domestic, long-haul (≥50 km)", "#eb6834", None, "domestic"),
    ("L_int_own", "International, own hauliers", "#1baf7a", None, "international"),
    ("L_int_foreign", "International, foreign hauliers", "#eda100", None, "international"),
    ("int_unsplit", "International, not split by haulier", SURFACE, "////", "international"),
]


def prepare(levels: pd.DataFrame) -> pd.DataFrame:
    """Add the not-split international segment and filled markers; convert to bn vkm."""
    df = levels.copy()
    split = df["L_int_own"].fillna(0) + df["L_int_foreign"].fillna(0)
    no_split = df["L_int_own"].isna() & df["L_int_foreign"].isna()
    df["int_unsplit"] = np.where(no_split, df["L_int"], df["L_int"] - split)
    df["int_unsplit"] = df["int_unsplit"].where(df["int_unsplit"] > 1e-9 * df["total"].abs().clip(lower=1))
    df["filled_domestic"] = (df["filled_L_dom"] != "observed") | (df["filled_alpha_E"] != "observed")
    df["filled_international"] = df["filled_L_int"] != "observed"
    for column, *_ in SEGMENTS:
        df[column] = df[column] / 1e9
    df["total"] = df["total"] / 1e9
    return df


def country_order(df: pd.DataFrame, reference_year: int) -> list[str]:
    """Countries by reference-year total, largest first; all-missing ones last."""
    ref = df[df["year"] == reference_year].set_index("country_id")["total"]
    ref = ref.reindex(sorted(df["country_id"].unique()))
    return list(ref.sort_values(ascending=False, na_position="last").index)


def draw_panel(ax, data: pd.DataFrame, years: list[int]) -> None:
    """Stacked bars for one country; filled segments lighter."""
    data = data.set_index("year").reindex(years)
    x = np.arange(len(years))
    bottom = np.zeros(len(years))
    for column, _, color, hatch, part in SEGMENTS:
        values = data[column].fillna(0).to_numpy()
        if not values.any():
            continue
        filled = data[f"filled_{part}"].fillna(False).to_numpy(dtype=bool)
        for mask, alpha in [(~filled, 1.0), (filled, FILLED_ALPHA)]:
            if not (values * mask).any():
                continue
            ax.bar(
                x[mask], values[mask], bottom=bottom[mask], width=0.78, color=color, hatch=hatch, alpha=alpha,
                edgecolor=SURFACE if hatch is None else MUTED, linewidth=0.5 if hatch is None else 0.4,
            )  # fmt: skip
        bottom += values
    ax.set_ylim(0, (bottom.max() if bottom.max() > 0 else 1.0) * 1.08)
    ax.set_xlim(-0.6, len(years) - 0.4)


def style_axis(ax, years: list[int]) -> None:
    """Recessive axes: no box, light horizontal grid, yearly or five-yearly ticks."""
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6, colors=TEXT_2, length=0)
    step = 1 if len(years) <= 6 else 5
    ticks = [i for i, y in enumerate(years) if y % step == 0]
    ax.set_xticks(ticks, [str(years[i]) for i in ticks])
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))


def plot(
    levels: pd.DataFrame, reference_year: int, output_path: str, ncols: int = 6
) -> None:
    """Write the small-multiples figure."""
    df = prepare(levels)
    years = sorted(df["year"].unique())
    countries = country_order(df, reference_year)
    nrows = math.ceil(len(countries) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.15 * ncols, 1.55 * nrows + 1.2),
        facecolor=SURFACE,
        squeeze=False,
    )
    for ax, country in zip(axes.flat, countries):
        data = df[df["country_id"] == country]
        style_axis(ax, years)
        draw_panel(ax, data, years)
        ref = data.loc[data["year"] == reference_year, "total"]
        all_filled = data["filled_domestic"].all() and data["filled_L_dom"].ne("observed").all()
        label = f"{ref.iloc[0]:.2f} bn in {reference_year}" + (" (filled)" if all_filled else "")
        ax.set_title(
            country, loc="left", fontsize=8, color=TEXT, fontweight="bold", pad=3
        )
        ax.text(
            1.0,
            1.02,
            label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color=TEXT_2,
        )
    for ax in axes.flat[len(countries) :]:
        ax.set_visible(False)

    handles = [
        Patch(facecolor=color, edgecolor=MUTED if hatch else color, hatch=hatch, linewidth=0.4, label=label)
        for _, label, color, hatch, _ in SEGMENTS
    ]
    handles.append(Patch(facecolor=MUTED, alpha=FILLED_ALPHA, edgecolor="none", label="Lighter: filled, not observed"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.45 / fig.get_figheight()),
               ncol=3, frameon=False, fontsize=7, labelcolor=TEXT, handlelength=1.2, columnspacing=1.5)  # fmt: skip
    fig.text(0.01, 1 - 0.12 / fig.get_figheight(), "Heavy-duty vehicle-km by component, after gap filling",
             fontsize=11, color=TEXT, fontweight="bold", va="top")  # fmt: skip
    fig.text(0.01, 1 - 0.32 / fig.get_figheight(),
             "Billion vehicle-km per year on each country's territory. Each panel has its own scale; "
             f"countries ordered by {reference_year} total. Filled: own series extended, or from neighbours "
             "(domestic per GDP in PPS, international per domestic, long-haul share).",
             fontsize=7.5, color=TEXT_2, va="top")  # fmt: skip
    fig.tight_layout(
        rect=(0, 0, 1, 1 - 1.0 / fig.get_figheight()), h_pad=1.2, w_pad=0.8
    )
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    plot(
        levels=pd.read_parquet(snakemake.input.levels),  # noqa: F821
        reference_year=snakemake.params.reference_year,  # noqa: F821
        output_path=snakemake.output.figure,  # noqa: F821
    )
