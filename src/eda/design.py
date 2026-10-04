import numpy as np
import pandas as pd
from data_handling import DataWriter
from eda import eda_common, eda_plots

def design(
        df: pd.DataFrame,
        target: str,
        output_cols: list,
        vary_col: str,
        n_groups_plot: int = 12,
        data_writer: DataWriter = None,
        generate_output: bool = True,
        verbose: bool = True
    ) -> None:
    """Analyze repeated configurations, target variability, and parameter families.

    Configurations are defined by identical values of all input columns.
    Families contain configurations that share all inputs except vary_col,
    allowing the target response to be compared as this input changes.

    For configurations with multiple runs, compute the mean target, relative
    standard deviation, and variance of the base-10 logarithm of the target.
    The relative standard deviation is the population standard deviation
    divided by the mean.

    Estimate an approximate R-squared ceiling in log10 space as:

        1 - weighted_noise_variance / total_log_variance

    The noise variance is the average within-configuration log variance,
    weighted by the number of runs. The total variance is computed across
    all runs belonging to repeated configurations. Both use ddof=0.

    This estimate assumes that differences between repeated runs represent
    noise that cannot be explained by the available inputs. It is not a
    guaranteed upper bound on test-set R-squared. If no repeated configurations
    exist, the estimate is None.

    Optionally print summary statistics, save the noise table and design
    summary, and generate target curves grouped by parameter family.

    Args:
        df (pd.DataFrame): Input dataset containing input and output columns.
        target (str): Target column to analyze.
        output_cols (list): Output columns excluded when identifying inputs.
        vary_col (str): Input column allowed to vary within each family and
            used as the horizontal axis in the family plot.
        n_groups_plot (int): Number of families requested for plotting.
            Defaults to 12.
        data_writer (DataWriter, optional): Writer used to save tables,
            results, and figures. If None, a writer is created when
            generate_output is True.
        generate_output (bool): Whether to save results and generate the
            family plot. Defaults to True.
        verbose (bool): Whether to print configuration counts, variability
            statistics, and the estimated R-squared ceiling. Defaults to True.

    Returns:  
        None: Results are printed or saved rather than returned.  

    Notes:  
        Target values must be strictly positive for the logarithmic analysis.  
        The R-squared calculation does not explicitly handle zero total  
        log variance.  
    """  
    cleaned = eda_common.clean_data(df)
    input_cols = eda_common.get_input_columns(cleaned, output_cols)

    family_cols = [] # vsetky vstupy okrem vary_col
    for col in input_cols:  
        if col != vary_col:  
            family_cols.append(col)  

    cleaned['config_id'] = cleaned.groupby(input_cols, sort=False).ngroup() # ngroup(): unikatnej kombinacii hodnot v stlpcoch priradi jedno cislo
    cleaned['family_id'] = cleaned.groupby(family_cols, sort=False).ngroup()

    runs_per_config = cleaned.groupby('config_id').size() 
    repeated_ids = runs_per_config[runs_per_config > 1].index # Opakovane konfiguracie = rovnake vstupy viackrat > 1
    repeated = cleaned[cleaned['config_id'].isin(repeated_ids)].copy()
    repeated['log_target'] = np.log10(repeated[target])

    grouped = repeated.groupby('config_id')
    noise = pd.DataFrame({
        'runs': grouped.size(),
        'mean': grouped[target].mean(),
        'relative_std': grouped[target].std(ddof=0) / grouped[target].mean(),
        'log_variance': grouped['log_target'].var(ddof=0),
    })

    if len(noise) > 0:
        noise_variance = (noise['log_variance'] * noise['runs']).sum() / noise['runs'].sum()
        total_variance = repeated['log_target'].var(ddof=0) # ddof=0 sucet stvorcov odchylok od priemeru sa deli poctom hodnot
        r2_ceiling = round(float(1 - noise_variance / total_variance), 3)
    else:
        r2_ceiling = None

    results = {
        'unique_configs': int(cleaned['config_id'].nunique()),
        'repeated_configs': int(len(repeated_ids)),
        'families': int(cleaned['family_id'].nunique()),
        'vary_col': vary_col,
        'r2_ceiling_log10': r2_ceiling
    }

    if verbose:
        print(f"Unique configurations: {results['unique_configs']}")
        print(f"Repeated configurations: {results['repeated_configs']}")
        print(f"Families (without '{vary_col}'): {results['families']}")
        print()
        print("Relative std of target within repeated runs:")
        print(noise['relative_std'].describe().round(4))
        print()
        print(f"R2 ceiling (log10 scale): {r2_ceiling}")

    if generate_output:
        if data_writer is None:
            data_writer = DataWriter()
        data_writer.write_data(noise, filename="03_noise_repeated_configs", subfolder="eda")
        data_writer.write_data(results, filename="03_design", subfolder="eda")

        figure = eda_plots.plot_lines_by_group(cleaned, vary_col, target, 'family_id', n_groups=n_groups_plot)
        data_writer.write_data(figure, filename="03_curves_families", subfolder="eda/figures")
        print()

    return None
