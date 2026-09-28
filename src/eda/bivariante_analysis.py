import pandas as pd

import utils.graphs as graphs
from utils.constants import TARGETS, FIG_DIR, REPORT_DIR, TARGET

def bivariante_analysis(df: pd.DataFrame, generate_output: bool = True, verbose: bool = True) -> None:
    """
    
    """
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_cols).drop_duplicates().reset_index(drop=True) # Vyhodi konstanty a duplicity, resetuje index na default ciselny index.

    INPUTS = [col for col in df.columns if col not in TARGETS]

    ALL_NUM = INPUTS + TARGETS
    graphs.plot_corr_heatmap(df, ALL_NUM, 'pearson',
                        save_path=f"{FIG_DIR}/03a_korelacie_pearson.png")
    graphs.plot_corr_heatmap(df, ALL_NUM, 'spearman',
                        save_path=f"{FIG_DIR}/03a_korelacie_spearman.png")

    cmp = pd.DataFrame({
        'pearson': df[ALL_NUM].corr(method='pearson')[TARGET],
        'spearman': df[ALL_NUM].corr(method='spearman')[TARGET],
    }).drop(TARGET).sort_values('spearman')
    print(cmp.round(3))

    return None