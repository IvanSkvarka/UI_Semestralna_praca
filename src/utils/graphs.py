import matplotlib.pyplot as plt
import seaborn as sns    
import math
import numpy as np
from pathlib import Path
import os

def set_style():
    """
    
    """
    sns.set_theme(style="whitegrid")

def _make_grid(n_plots, ncols, cell_w, cell_h):
    """
    """
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_plots:]:      # prázdne bunky na konci mriežky skryjeme
        ax.set_visible(False)
    return fig, axes

def _finish(fig, title, save_path, dpi=130, title_y=1.0, bbox=None):
    """
    
    """
    if title:
        fig.suptitle(title, fontsize=14, y=title_y)
    fig.tight_layout()
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox)
        plt.close(fig)
    return fig

def plot_histograms(df, cols, ncols=3, bins=30, kde=True, title="Rozloženie premenných", save_path=None):
    """

    """
    fig, axes = _make_grid(len(cols), ncols, cell_w=5, cell_h=3.6)
    for ax, col in zip(axes, cols):
        sns.histplot(df[col], kde=kde, bins=bins, ax=ax, color="steelblue")
        ax.set_title(f"{col}  (skew={df[col].skew():.2f}, unikátnych={df[col].nunique()})",
                     fontsize=10)
        ax.set_xlabel("")
    return _finish(fig, title, save_path)

def plot_target_raw_vs_log(df, targets, bins=40, title="Ciele: pôvodná vs. logaritmická škála", save_path=None):
    """

    """
    if (df[targets] <= 0).any().any():
        raise ValueError("log10 vyžaduje striktne kladné hodnoty vo všetkých cieľoch.")
 
    fig, axes = plt.subplots(len(targets), 2, figsize=(12, 3.2 * len(targets)),
                             squeeze=False)
    for i, col in enumerate(targets):
        log_col = np.log10(df[col])
        sns.histplot(df[col], kde=True, bins=bins, ax=axes[i, 0], color="indianred")
        axes[i, 0].set_title(f"{col} - pôvodná škála (skew={df[col].skew():.2f})", fontsize=10)
        sns.histplot(log_col, kde=True, bins=bins, ax=axes[i, 1], color="seagreen")
        axes[i, 1].set_title(f"log10({col}) (skew={log_col.skew():.2f})", fontsize=10)
        axes[i, 0].set_xlabel("")
        axes[i, 1].set_xlabel("")
    return _finish(fig, title, save_path)

def plot_boxplots(df, cols, ncols=4, log_scale=False, title="Boxploty - odľahlé hodnoty", save_path=None):
    """

    """
    fig, axes = _make_grid(len(cols), ncols, cell_w=4, cell_h=2.8)
    for ax, col in zip(axes, cols):
        sns.boxplot(x=df[col], ax=ax, color="lightsteelblue", fliersize=3)
        if log_scale:
            ax.set_xscale("log")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    return _finish(fig, title, save_path)
 
 
def plot_discrete_counts(df, cols, title="Početnosti diskrétnych premenných", save_path=None):
    """

    """
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        counts = df[col].value_counts().sort_index()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="slateblue")
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(col)
        ax.set_xlabel("hodnota")
        ax.set_ylabel("počet")
    return _finish(fig, title, save_path, title_y=1.03, bbox="tight")











def plot_corr_heatmap(df, cols, method="pearson", annot=True,
                      title=None, save_path=None):
    """
    Heatmapa korelačnej matice (zobrazený len dolný trojuholník - matica je
    symetrická, horný by len duplikoval informáciu).
 
    method:
      'pearson'  - meria LINEÁRNU závislosť; citlivý na outliery a zošikmenie.
      'spearman' - meria MONOTÓNNU závislosť (počíta sa z poradí hodnôt);
                   robustný voči outlierom a odhalí aj nelineárne (napr.
                   mocninné) vzťahy. Preto v práci uvádzame oba a porovnávame.
    Fixná škála -1..1 a divergujúca paleta: červená = kladná, modrá = záporná.
    """
    corr = df[cols].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    size = max(7, 0.62 * len(cols))
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                annot=annot, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, square=True, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title(title or f"Korelačná matica ({method})", fontsize=13)
    return _finish(fig, None, save_path)
 
 
def plot_scatter_vs_target(df, x_cols, target, hue=None, ncols=3,
                           logx="auto", logy=False, alpha=0.5,
                           title=None, save_path=None):
    """
    Mriežka scatter plotov: každý panel = jeden vstup (os x) vs. cieľ (os y).
 
    logx: True / False / 'auto'. 'auto' použije log os len pre stĺpce so
          striktne kladnými hodnotami (log nulového/záporného čísla neexistuje).
    logy: True -> logaritmická os cieľa.
    hue:  názov stĺpca, podľa ktorého sa body farebne odlíšia (napr. n_impeller).
 
    Log-log súradnice sú kľúčové pri fyzikálnych veličinách: mocninný vzťah
    y = a * x^b sa tam zobrazí ako PRIAMKA so smernicou b, kým v lineárnych
    súradniciach vyzerá ako zakrivená (alebo dokonca nekorelovaná) mračno bodov.
    """
    fig, axes = _make_grid(len(x_cols), ncols, cell_w=5, cell_h=4)
    for ax, col in zip(axes, x_cols):
        sns.scatterplot(data=df, x=col, y=target, hue=hue, ax=ax, s=18,
                        alpha=alpha, palette="viridis" if hue else None,
                        color=None if hue else "steelblue", edgecolor=None,
                        legend=(ax is axes[0]) if hue else False)
        use_logx = (df[col] > 0).all() if logx == "auto" else bool(logx)
        if use_logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        r = df[[col, target]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{col}  (Spearman={r:.2f})", fontsize=10)
    return _finish(fig, title or f"{target} vs. vstupy", save_path)
 
 
def plot_pairplot(df, cols, hue=None, title="Pairplot", save_path=None):
    """
    Pairplot (seaborn): scatter ploty všetkých dvojíc + rozdelenie na diagonále.
    Zobrazený len dolný trojuholník (corner=True), aby graf zostal čitateľný.
    Odporúča sa len pre malú podmnožinu (max ~6 stĺpcov) - inak je nečitateľný.
    """
    data = df[cols + ([hue] if hue and hue not in cols else [])]
    grid = sns.pairplot(data, hue=hue, corner=True, diag_kind="kde",
                        plot_kws={"s": 14, "alpha": 0.5, "edgecolor": None},
                        palette="viridis" if hue else None, height=2.2)
    grid.figure.suptitle(title, fontsize=14, y=1.02)
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        grid.figure.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(grid.figure)
    return grid