import numpy as np
import pandas as pd

from data_handling import DataWriter
from eda import eda_common, eda_plots

def overview(
        df: pd.DataFrame,
        output_columns: list,
        max_discrete: int = 2,
        data_writer: DataWriter = None,
        generate_output: bool = True,
        LaTeX_graphs: bool = False,
        verbose: bool = True
    ) -> None:
    """ Summarize dataset quality and the distributions of inputs and outputs.

    Inspect the raw dataset for column types, missing values, unique values,
    constant columns, and duplicate rows. Compute distribution statistics
    using the cleaned dataset.

    Classify numeric inputs as discrete if their number of unique values is
    at most max_discrete, and as continuous otherwise. For discrete inputs,
    count occurrences of each value. For continuous inputs, compute
    descriptive statistics, skewness, and the percentage of values outside
    the interval [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].

    For output columns containing only strictly positive values, compare
    skewness before and after a base-10 logarithmic transformation.

    Optionally print the dataset summary and continuous-input statistics,
    save the generated tables and summary, and plot output distributions
    in their original and log10 scales.

    Args:  
        df (pd.DataFrame): Input dataset containing input and output columns.
        output_columns (list): Output columns excluded when identifying
            inputs and considered for the output distribution analysis.
        max_discrete (int): Maximum number of unique values for a numeric
            input to be classified as discrete. Defaults to 2.
        data_writer (DataWriter, optional): Writer used to save tables,
            results, and figures. If None, a writer is created when
            generate_output is True.
        generate_output (bool): Whether to save results and generate the
            output distribution plot. Defaults to True.
        LaTeX_graphs (bool): Whether to use LaTeX in graphs. Defaults to False.
        verbose (bool): Whether to print the dataset summary, column
            information, and continuous-input statistics. Defaults to True.

    Returns:
        None: Results are printed or saved rather than returned.

    Notes:
        The discrete/continuous classification is based only on the number
        of observed unique values, not on the semantic meaning of a column.
        Values flagged by the IQR rule are counted but not removed.
        Outputs containing non-positive or missing values are excluded from
        the logarithmic comparison.
    """
    column_info = pd.DataFrame({
        'type': df.dtypes.astype(str),
        'missing': df.isna().sum(),
        'unique_values': df.nunique()
    })

    cleaned = eda_common.clean_data(df)
    input_cols = eda_common.get_numeric_columns(cleaned, eda_common.get_input_columns(cleaned, output_columns))

    discrete_cols = []
    continuous_cols = []
    for col in input_cols:
        if cleaned[col].nunique() <= max_discrete:
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)

    input_stats = cleaned[continuous_cols].describe().T
    input_stats['skew'] = cleaned[continuous_cols].skew() # Nesumernost dat

    discrete_inputs = []
    for col in discrete_cols:
        value_counts = cleaned[col].value_counts().sort_index()
        for value, count in value_counts.items():
            discrete_inputs.append({
                'Column': col,
                'Value': value,
                'Count': count
            })
    discrete_inputs = pd.DataFrame(discrete_inputs)

    outlier_percent = [] # podiel outlierov (pravidlo 1.5 * IQR)
    for col in continuous_cols:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1 # Vypocet medzikvartiloveho rozpatia (IQR)
        is_outlier = (cleaned[col] < q1 - 1.5 * iqr) | (cleaned[col] > q3 + 1.5 * iqr)
        outlier_percent.append(100 * is_outlier.mean())
    input_stats['outliers_%'] = outlier_percent

    positive_outputs = [] # Porovnanie asymetrie vystupov pred a po log10 -> log iba pre kladne
    for col in output_columns:
        if (cleaned[col] > 0).all():
            positive_outputs.append(col)

    output_skew = pd.DataFrame({
        'raw': cleaned[positive_outputs].skew(),
        'log10': np.log10(cleaned[positive_outputs]).skew(),
    })

    results = {
        'rows_raw': int(df.shape[0]),
        'columns_raw': int(df.shape[1]),
        'non_numeric_cols': df.select_dtypes(exclude='number').columns.tolist(),
        'missing_total': int(df.isna().sum().sum()),
        'constant_cols': eda_common.get_constant_columns(df),
        'identical_rows': int(df.duplicated().sum()),
        'rows_clean': int(cleaned.shape[0]),
        'columns_clean': int(cleaned.shape[1]),
        'discrete_inputs': discrete_cols,
        'continuous_inputs': continuous_cols
    }

    if verbose:
        print(f"Raw data: {results['rows_raw']} rows, {results['columns_raw']} columns.")
        print(f"Non-numeric columns: {results['non_numeric_cols']}")
        print(f"Missing values: {results['missing_total']}")
        print(f"Constant columns: {results['constant_cols']}")
        print(f"Identical rows: {results['identical_rows']}")
        print(f"After cleaning: {results['rows_clean']} rows, {results['columns_clean']} columns.")
        print()
        print(column_info)
        print()
        print("Continuous inputs:")
        print(input_stats.round(3))
        print()

    if generate_output:
        if data_writer is None:
            data_writer = DataWriter()
        data_writer.write_data(column_info, filename="01_column_info", subfolder="eda")
        data_writer.write_data(input_stats.T, filename="01_input_statistics", subfolder="eda")
        data_writer.write_data(discrete_inputs, filename="01_discrete_inputs", subfolder="eda")
        data_writer.write_data(output_skew, filename="01_output_skew", subfolder="eda")
        data_writer.write_data(results, filename="01_overview", subfolder="eda")

        figure = eda_plots.plot_target_raw_vs_log(cleaned, positive_outputs, LaTeX_graphs)
        data_writer.write_data(figure, LaTeX_graphs, filename="01_outputs_raw_vs_log", subfolder="eda/figures")
        print()
        
    return None