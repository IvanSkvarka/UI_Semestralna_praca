import pandas as pd
import numpy as np

import utils.graphs as graphs
from utils.constants import TARGETS, DISCRETE, FIG_DIR, REPORT_DIR

def univariate_analysis(df: pd.DataFrame, generate_output: bool = True, verbose: bool = True) -> None:
    """
    
    """
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_cols).drop_duplicates().reset_index(drop=True) # Vyhodi konstanty a duplicity, resetuje index na default ciselny index.

    discrete_cols = [col for col in DISCRETE if col in df.columns]
    continuous_cols = [col for col in df.columns if col not in TARGETS + DISCRETE]

    num_cols = continuous_cols + TARGETS
    stats = df[num_cols].describe().T # Opisuje DataFrame, vstavana funkcia
    stats['skew'] = df[num_cols].skew() # Zosikmenie: 0 = symetricke, >1 = silne pravostranne
    stats['kurtosis'] = df[num_cols].kurtosis()  # Strmost: Vysoke = silne chvosty - viac outlierov

    def count_outliers_iqr(s: pd.Series) -> pd.Series:
        """
        IQR pravidlo pre outliery: hodnota je odľahlá, ak leží mimo
        [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Zvolil som ho pred z-skóre, pretože
        z-skóre predpokladá približne normálne rozdelenie - a naše dáta sú
        výrazne zošikmené, takže by z-skóre nesprávne označilo veľa
        "normálnych" hodnôt z dlhého chvosta ako outliery.
        """
        q1, q3 = s.quantile([0.25, 0.75]) # Vypocita kvartily
        iqr = q3 - q1
        return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    
    stats['n_outliers_IQR'] = [count_outliers_iqr(df[col]) for col in num_cols]
    stats['pct_outliers'] = (stats['n_outliers_IQR'] / len(df) * 100).round(1)

    log_skew = pd.DataFrame({ # Ukážeme, ako log transformácia zošikmenie cieľov zníži (dôvod, prečo ju použijeme).
        'skew_orig': df[TARGETS].skew(),
        'skew_log10': np.log10(df[TARGETS]).skew()
    }).round(3)

    if verbose:
        print(stats[
            ['min', '50%', 'max', 'mean', 'std', 'skew', 'kurtosis','n_outliers_IQR', 'pct_outliers']].round(4)) # 50% = median
        print("\nLogarithmic (log) skewness of targets after log10 transformation:")
        print(log_skew)

    if generate_output:
        graphs.plot_histograms(df, continuous_cols, ncols=3,
                   title="Rozloženie spojitých vstupných premenných",
                   save_path=f"{FIG_DIR}/02b_histogramy_vstupy.png")
 
        graphs.plot_target_raw_vs_log(df, TARGETS,
                                save_path=f"{FIG_DIR}/02c_ciele_povodna_vs_log.png")
        
        graphs.plot_boxplots(df, continuous_cols + TARGETS, ncols=4,
                        title="Boxploty (pôvodná škála) - odľahlé hodnoty",
                        save_path=f"{FIG_DIR}/02d_boxploty.png")
        
        graphs.plot_discrete_counts(df, discrete_cols,
                                title="Početnosti diskrétnych vstupov",
                                save_path=f"{FIG_DIR}/02e_diskretne_pocetnosti.png")
        
        print(f"\nGrafy uložené do priečinka '{FIG_DIR}/'.")

    return None