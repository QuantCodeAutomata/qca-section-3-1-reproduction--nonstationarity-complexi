import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RECESSION_BARS = [
    (1990, 1991),   # Gulf War recession: Jul 1990 – Mar 1991
    (2001, 2001),   # Dot-com recession: Mar – Nov 2001
    (2007, 2009),   # GFC: Dec 2007 – Jun 2009
]

_WEALTH_COLORS = [
    ("ATOMS", "red"),
    ("32",    "blue"),
    ("128",   "cyan"),
    ("512",   "green"),
    ("CV",    "orange"),
]

_WINNER_COLORS = {
    "ridge_64": "steelblue",
    "rf_64":    "darkorange",
    "rf_all":   "green",
}


def _ensure_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not already exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _wealth_color(name: str) -> Optional[str]:
    for pattern, color in _WEALTH_COLORS:
        if pattern in name:
            return color
    return None


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a plain Markdown table (no external dependencies)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "|" + "|".join(" :--- " if df[c].dtype == object else " ---: " for c in cols) + "|"
    rows = []
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    index_header = f"| **{df.index.name or 'index'}** | " + " | ".join(str(c) for c in cols) + " |"
    index_sep    = "|" + " --- |" * (len(cols) + 1)
    index_rows   = []
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        index_rows.append(f"| **{idx}** | " + " | ".join(cells) + " |")
    return "\n".join([index_header, index_sep] + index_rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_annual_winner_counts(winner_counts_df: pd.DataFrame, output_path: str) -> None:
    """Bar chart: for each year, how many industries each model won.

    Parameters
    ----------
    winner_counts_df:
        DataFrame with year index and one column per model
        (expected column names: ``ridge_64``, ``rf_64``, ``rf_all``).
    output_path:
        Full path for the output PNG file.
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(14, 5))

    if winner_counts_df.empty:
        ax.set_title("Annual Model Winner Counts Across 17 Industries (Section 3.1)")
        plt.savefig(output_path, dpi=150)
        plt.close("all")
        return

    years = list(winner_counts_df.index)
    n_years = len(years)
    n_cols = len(winner_counts_df.columns)
    bar_width = 0.8 / max(n_cols, 1)
    x = np.arange(n_years)

    for i, col in enumerate(winner_counts_df.columns):
        offsets = x + i * bar_width - (n_cols - 1) * bar_width / 2
        color = _WINNER_COLORS.get(col)
        ax.bar(offsets, winner_counts_df[col].values, width=bar_width,
               label=col, color=color)

    # NBER recession shading (positions are 0-based bar indices)
    year_to_pos = {yr: pos for pos, yr in enumerate(years)}
    for rec_start, rec_end in _RECESSION_BARS:
        pos_start = year_to_pos.get(rec_start)
        pos_end   = year_to_pos.get(rec_end)
        if pos_start is None and pos_end is None:
            continue
        xs = pos_start if pos_start is not None else pos_end
        xe = pos_end   if pos_end   is not None else pos_start
        ax.axvspan(xs - 0.5, xe + 0.5, color="lightgray", alpha=0.5, zorder=0,
                   label="_nolegend_")

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Industries")
    ax.set_title("Annual Model Winner Counts Across 17 Industries (Section 3.1)")
    ax.legend()
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close("all")


def plot_annual_r2_by_industry(annual_r2_df: pd.DataFrame, output_path: str) -> None:
    """Line plot of annual OOS R² averaged across industries.

    Parameters
    ----------
    annual_r2_df:
        DataFrame with year index and one column per model (or per industry).
        Industry-averaged R² is plotted when multiple columns are present.
    output_path:
        Full path for the output PNG file.
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(12, 5))

    if annual_r2_df.empty:
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("OOS R²")
        ax.set_title("Annual OOS R² (Industry Average)")
        plt.savefig(output_path, dpi=150)
        plt.close("all")
        return

    for col in annual_r2_df.columns:
        ax.plot(annual_r2_df.index, annual_r2_df[col].values,
                marker="o", markersize=3, linewidth=1.5, label=col)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("OOS R²")
    ax.set_title("Annual OOS R² (Industry Average)")
    ax.legend()
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close("all")


def plot_recession_table(recession_df: pd.DataFrame, output_path: str) -> None:
    """Heatmap of OOS R² values across methods (rows) and evaluation windows (columns).

    Color scale: red (negative) → white (zero) → green (positive).

    Parameters
    ----------
    recession_df:
        DataFrame with methods as index and evaluation windows as columns.
    output_path:
        Full path for the output PNG file.
    """
    _ensure_dir(output_path)

    nrows = max(len(recession_df.index) + 1, 4)
    ncols = max(len(recession_df.columns) + 1, 4)
    fig_h = max(nrows * 0.6, 4)
    fig_w = max(ncols * 1.4, 8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if recession_df.empty:
        ax.set_title("Average OOS R² by Method and Evaluation Window")
        plt.savefig(output_path, dpi=150)
        plt.close("all")
        return

    sns.heatmap(
        recession_df,
        annot=True,
        fmt=".3f",
        center=0,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Average OOS R² by Method and Evaluation Window")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close("all")


def plot_wealth_paths(
    wealth_paths: Dict[str, pd.Series],
    output_path: str,
    title: str = "Cumulative Wealth Paths",
) -> None:
    """Line plot of log(wealth) over time for each strategy.

    Parameters
    ----------
    wealth_paths:
        Mapping from strategy name to a Series of cumulative wealth values
        (strictly positive).  Expected keys include substrings used for
        automatic colour assignment: ``ATOMS``, ``32``, ``128``, ``512``,
        ``CV``.
    output_path:
        Full path for the output PNG file.
    title:
        Plot title.
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(12, 6))

    if not wealth_paths:
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Wealth")
        ax.set_title(title)
        plt.savefig(output_path, dpi=150)
        plt.close("all")
        return

    for name, series in wealth_paths.items():
        safe_vals = series.values.copy().astype(float)
        safe_vals[safe_vals <= 0] = np.nan
        log_w = np.log(safe_vals)
        color = _wealth_color(name)
        kwargs: dict = {"label": name, "linewidth": 1.5}
        if color is not None:
            kwargs["color"] = color
        ax.plot(series.index, log_w, **kwargs)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Wealth")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close("all")


def plot_terminal_wealth_excess(excess_df: pd.DataFrame, output_path: str) -> None:
    """Bar chart of ATOMS terminal wealth excess ratio vs each baseline.

    Parameters
    ----------
    excess_df:
        DataFrame with baseline method names as index and at least one
        numeric column of excess-ratio values.
    output_path:
        Full path for the output PNG file.
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(9, 5))

    if excess_df.empty:
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Baseline Method")
        ax.set_ylabel("Average Terminal Wealth Excess Ratio")
        ax.set_title("ATOMS Terminal Wealth Excess vs Baselines")
        plt.savefig(output_path, dpi=150)
        plt.close("all")
        return

    numeric_cols = excess_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        plt.savefig(output_path, dpi=150)
        plt.close("all")
        return

    values = excess_df[numeric_cols].mean(axis=1)
    labels = [str(lbl) for lbl in values.index]
    colors = ["steelblue" if v >= 0 else "salmon" for v in values.values]
    ax.bar(range(len(values)), values.values, tick_label=labels, color=colors)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Baseline Method")
    ax.set_ylabel("Average Terminal Wealth Excess Ratio")
    ax.set_title("ATOMS Terminal Wealth Excess vs Baselines")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close("all")


def save_all_results_markdown(results_dict: dict, output_dir: str = "results") -> None:
    """Write a Markdown summary of experiment results to ``<output_dir>/RESULTS.md``.

    Parameters
    ----------
    results_dict:
        Mapping from experiment name to result data.  Values may be:
        - ``pd.DataFrame`` — rendered as a Markdown table.
        - ``dict``         — rendered as a two-column key/value table.
        - anything else   — converted to string.
    output_dir:
        Directory in which ``RESULTS.md`` is written (created if absent).
    """
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "RESULTS.md")

    with open(md_path, "w") as fh:
        fh.write("# Experiment Results\n\n")
        fh.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fh.write("---\n\n")

        for exp_name, exp_data in results_dict.items():
            fh.write(f"## {exp_name}\n\n")

            if isinstance(exp_data, pd.DataFrame):
                if exp_data.empty:
                    fh.write("*No data.*\n\n")
                else:
                    fh.write(_df_to_markdown_table(exp_data))
                    fh.write("\n\n")

            elif isinstance(exp_data, dict):
                fh.write("| Metric | Value |\n")
                fh.write("| ------ | ----- |\n")
                for k, v in exp_data.items():
                    formatted = f"{v:.4f}" if isinstance(v, float) else str(v)
                    fh.write(f"| {k} | {formatted} |\n")
                fh.write("\n")

            else:
                fh.write(f"{exp_data}\n\n")

        fh.write("---\n\n")
        fh.write("*End of results.*\n")
