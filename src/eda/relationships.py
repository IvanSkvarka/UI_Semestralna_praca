import numpy as np
import pandas as pd
from data_handling import DataWriter
from eda import eda_common, eda_plots

def relationships(
        df: pd.DataFrame,
        target: str,
        output_cols: list,
        threshold: float = 0.95,
        data_writer: DataWriter = None,
        generate_output: bool = True,
        verbose: bool = True
    ) -> None:
    """ Analyze correlations between inputs and the target and identify redundant inputs.

    Pearson correlation measures linear relationships, while Spearman
    correlation measures monotonic relationships that need not be linear.
    Values close to +1 or -1 indicate strong positive or negative
    relationships. Values close to 0 indicate weak relationships of the
    corresponding type. A weak correlation does not necessarily mean that
    a variable is not useful.

    Large differences between Pearson and Spearman correlations may  
    indicate nonlinear monotonic relationships or the influence of outliers.  

    Input pairs with an absolute Pearson correlation above the threshold
    are flagged as potentially redundant. Such relationships may indicate
    multicollinearity, which can reduce the stability and interpretability
    of linear regression coefficients.

    Optionally save correlation tables, flagged input pairs, a Spearman
    correlation heatmap, and input-versus-target scatter plots. The heatmap
    shows the overall relationship structure. Scatter plots help reveal
    curvature, saturation, outliers, operating regimes, changing variance,
    and relationships not captured by correlation coefficients.

    Args:
        df: Input dataset to clean and analyze.
        target: Target column used for correlation analysis and scatter plots.
        output_cols: Output column names, excluded from input selection.
        threshold: Absolute Pearson correlation threshold for flagging
            input pairs. Defaults to 0.95.
        data_writer: Writer used to save results and figures. If None,
            a DataWriter is created when output generation is enabled.
        generate_output: Whether to save results and generate plots.
            Defaults to True.
        verbose: Whether to print correlations and flagged input pairs.
            Defaults to True.

    Returns: 
        None.
    """
    cleaned = eda_common.clean_data(df)
    input_cols = eda_common.get_numeric_columns(cleaned, eda_common.get_input_columns(cleaned, output_cols))
    all_cols = input_cols + output_cols

    correlations = pd.DataFrame({
        'pearson': cleaned[all_cols].corr(method='pearson')[target], # Pearson (linearny vztah)
        'spearman': cleaned[all_cols].corr(method='spearman')[target], # Spearman (monotonny vztah)
    })
    correlations = correlations.drop(target)
    correlations['difference'] = (correlations['pearson'] - correlations['spearman']).abs()

    input_corr = cleaned[input_cols].corr().abs() # Multikolinearita: dvojice vstupov s |r| > threshold
    correlated_pairs = []
    for i in range(len(input_cols)):
        for j in range(i + 1, len(input_cols)): # j > i -> kazda dvojica iba raz
            col_a = input_cols[i]
            col_b = input_cols[j]
            r = input_corr.loc[col_a, col_b]
            if r > threshold:
                correlated_pairs.append(f"{col_a} <-> {col_b}: {r:.3f}")

    results = {
        'target': target,
        'threshold': threshold,
        'correlated_pairs': correlated_pairs,
    }

    if verbose:
        print(f"Correlations with '{target}':")
        print(correlations.sort_values('spearman').round(3))
        print()
        print(f"Input pairs with |r| > {threshold}: {correlated_pairs}")

    if generate_output:
        if data_writer is None:
            data_writer = DataWriter()
        data_writer.write_data(correlations, filename="02_correlations", subfolder="eda")
        data_writer.write_data(results, filename="02_relationships", subfolder="eda")

        heatmap = eda_plots.plot_corr_heatmap(cleaned, all_cols, 'spearman')
        data_writer.write_data(heatmap, filename="02_correlations_spearman", subfolder="eda/figures")

        scatter = eda_plots.plot_scatter_vs_target(cleaned, input_cols, target)
        data_writer.write_data(scatter, filename="02_scatter_inputs", subfolder="eda/figures")
        print()
        
    return None
