from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


def set_style() -> None:
    """Set a consistent plot style."""
    sns.set_theme(style="whitegrid")


def _set_size(width_pt=390.0, fraction=1, subplots=(1, 1)) -> list[float, float]:
    """Set optimal plot size."""
    fig_width_in = width_pt * fraction / 72.27
    
    golden_ratio = (5**.5 - 1) / 2
    
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    
    return (fig_width_in, fig_height_in)


def _finish(
    fig: Figure,
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 130
) -> Figure:
    """Adjust the layout and optionally save the figure.

    Args:
        fig: Figure to finalize.
        title: Overall figure title.
        save_path: Output path. None disables saving.
        dpi: Resolution of the saved image.

    Returns:
        Finalized figure. The figure is also closed when saved.
    """
    if title:
        fig.suptitle(title)

    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    _reset_graph_export()
    return fig

def _to_latex_label(text: str) -> str:
    if "_" in text:
        prefix, suffix = text.split("_", 1)
        p_style = f"\\mathrm{{{prefix}}}" if len(prefix) > 1 else prefix
        s_style = f"\\mathrm{{{suffix}}}"
        return f"${p_style}_{{{s_style}}}$"
    else:
        return f"$\\mathrm{{{text}}}$" if len(text) > 1 else f"${text}$"


def _set_graph_export() -> None:
    """Set graph to TeX style."""
    plt.rcParams.update({
        "pgf.preamble": "\n".join([
            r"\usepackage{mathptmx}",
        ]),
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "ptm"],
        "font.size": 12,
        "xtick.labelsize": 12 * 0.8,
        "ytick.labelsize": 12 * 0.8,
        "legend.fontsize": 12 * 0.6,
        "pgf.rcfonts": False,
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
    })

    return None

def _reset_graph_export() -> None:
    "Reset graph style to default."
    plt.rcParams.update(plt.rcParamsDefault)

    return None
    

def plot_target_raw_vs_log(
    df: pd.DataFrame,
    targets: list[str],
    *args,
    bins: int = 40,
    title: str | None = "Targets: original vs. logarithmic scale",
) -> Figure:
    """Compare target histograms before and after log transformation.

    Args:
        df: Input data.
        targets: Names of the target columns.
        bins: Number of histogram bins.
        title: Overall figure title.

    Returns:
        Figure containing two histograms for each target.

    Raises:
        ValueError: If target columns contain nonpositive values.
    """
    set_style()
    if args[0]:
        _set_graph_export()

    if (df[targets] <= 0).any().any():
        raise ValueError("Log transformation requires positive values.")

    total_plots = 2 * len(targets)
    total_subplots = total_plots * ((5**.5 - 1) / 2) if total_plots > 3 else 1
    fig_width_in, fig_height_in = _set_size(subplots=(total_subplots, 2))

    fig, axes = plt.subplots(
        nrows=len(targets),
        ncols=2,
        figsize=(fig_width_in, fig_height_in),
        squeeze=False,
    )

    for i, col in enumerate(targets):
        sns.histplot(df[col], bins=bins, kde=True, ax=axes[i, 0])
        sns.histplot(np.log10(df[col]), bins=bins, kde=True, ax=axes[i, 1])

        axes[i, 0].set_title(_to_latex_label(col))
        axes[i, 1].set_title(f"$\\log_{{10}}$({_to_latex_label(col)})")

        axes[i, 0].set_xlabel(_to_latex_label(col))
        axes[i, 1].set_xlabel(f"$\\log_{{10}}$({_to_latex_label(col)})")


    return _finish(fig, None if args[0] else title)

# TODO: Opravit, tak aby kazda fun prijimala argument LaTeX
def plot_corr_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    method: Literal["pearson", "spearman", "kendall"] = "spearman",
    title: str | None = None
) -> Figure:
    """Plot the lower triangle of a correlation matrix.

    Args:
        df: Input data.
        cols: Names of the columns to compare.
        method: Correlation method.
        title: Figure title. None uses the default title.

    Returns:
        Figure containing the correlation heatmap.
    """
    set_style()

    corr = df[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        data=corr,
        mask=np.triu(np.ones_like(corr, dtype=bool), k=1), # vytvori horny trojuholnik plny jednotiek od k=1 diagonaly, na ktorych miestach budu skryte (mask) hodnoty
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())

    orig_x_labels = [t.get_text() for t in ax.get_xticklabels()]
    orig_y_labels = [t.get_text() for t in ax.get_yticklabels()]

    new_x_labels = [_to_latex_label(l) for l in orig_x_labels]
    new_y_labels = [_to_latex_label(l) for l in orig_y_labels]

    ax.set_xticklabels(new_x_labels, rotation=90, fontsize=10)
    ax.set_yticklabels(new_y_labels, rotation=0, fontsize=10)

    return _finish(fig, title or f"Correlations ({method})")


def plot_scatter_vs_target(
    df: pd.DataFrame,
    x_cols: list[str],
    target: str,
    hue: str | None = None,
    n_cols: int = 2,
    logx: bool | Literal["auto"] = "auto",
    logy: bool = True,
    title: str | None = None
) -> Figure:
    """Plot scatter plots of input columns against one target.

    Args:
        df: Input data.
        x_cols: Column names for the x-axes.
        target: Column name for the y-axis.
        hue: Optional column determining point colors.
        n_cols: Number of plots per row.
        logx: Use a logarithmic x-axis; "auto" enables it for positive columns.
        logy: Whether to use a logarithmic y-axis.
        title: Overall figure title. None uses the default title.

    Returns:
        Figure containing a grid of scatter plots.
    """
    set_style()

    n_rows = (len(x_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel() # Splostene do 1-D pola (n, )

    # TODO: Zistit co robia parametre 
    for i, col in enumerate(x_cols):
        ax = axes[i]
        sns.scatterplot(
            data=df,
            x=col,
            y=target,
            hue=hue,
            ax=ax,
            s=18,
            alpha=0.55,
            legend=bool(hue) and i == 0,
        )

        use_logx = (df[col] > 0).all() if logx == "auto" else logx
        if use_logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        ax.set_xlabel(_to_latex_label(col))
        ax.set_ylabel(_to_latex_label(target))
        ax.set_title(f"{_to_latex_label(target)} vs. {_to_latex_label(col)}")

    for ax in axes[len(x_cols):]:
        ax.set_visible(False)

    return _finish(fig, title or f"{target} vs. inputs")


def plot_lines_by_group(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_col: str,
    n_groups: int = 12,
    logx: bool = True,
    logy: bool = True,
    seed: int = 1,
    title: str | None = None
) -> Figure:
    """Plot lines for randomly selected groups.

    Args:
        df: Input data.
        x: Column name for the x-axis.
        y: Column name for the y-axis.
        group_col: Column identifying groups.
        n_groups: Maximum number of groups to select.
        logx: Whether to use a logarithmic x-axis.
        logy: Whether to use a logarithmic y-axis.
        seed: Random seed for reproducible selection.
        title: Figure title. None uses the default title.

    Returns:
        Figure containing one line for each selected group.
    """
    set_style()

    groups = df[group_col].unique()
    chosen = np.random.default_rng(seed).choice(
        groups,
        size=min(n_groups, len(groups)),
        replace=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for group in chosen:
        sub = df[df[group_col] == group].sort_values(x)
        ax.plot(sub[x], sub[y], "o-", markersize=4)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.set(xlabel=_to_latex_label(x), ylabel=_to_latex_label(y))

    return _finish(fig, title or f"{y} vs. {x}, {len(chosen)} random groups")
