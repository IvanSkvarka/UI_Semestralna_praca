import pandas as pd
from typing import Optional
from data_handling import DataWriter

def overview(
        df: pd.DataFrame,
        output_cols: Optional[list],
        generate_output: bool = True,
        verbose: bool = True
    ) -> None:
    """ Perform a statistical overview of a pandas DataFrame.
    
    This function isolates and analyzes the structural characteristics of a dataset.
    It aggregates per-column metrics including data types, missing values, distinct
    value counts, and internal duplication rates.
    
    Args:
        df (pd.DataFrame): The target DataFrame containing the dataset to analyze.
        output_cols (Optional[list]): A list of column names representing target or 
            output variables. These columns are dynamically excluded when calculating 
            duplicate combinations in the input configuration. If None or empty, 
            all columns are treated as input features.
        generate_output (bool, optional): If True, triggers the generation and 
            export of the overview metrics into an external file. Defaults to True.
        verbose (bool, optional): If True, prints a formatted summary table and 
            structural metadata directly to the standard output. Defaults to True.

    Returns:
        None: This function does not return any value; it performs in-place 
            printing and file generation operations.

    Raises:
        TypeError: If the input `df` argument is not a valid pandas DataFrame 
            or if `output_cols` is provided but is not a list.
        ValueError: If the provided DataFrame is completely empty.

    Examples:
        >>> import pandas as pd
        >>> data = {
        ...     'feature_A':,
        ...     'feature_B': ['X', 'Y', 'Y', 'Z'],
        ...     'target_val': [10.5, 20.0, 99.9, 40.2]
        ... }
        >>> sample_df = pd.DataFrame(data)
        >>> overview(sample_df, output_cols=['target_val'], generate_output=False, verbose=True)
        DataFrame size: 3 columns and 4 rows.
        <BLANKLINE>
                    dtypes  n_missing  n_unique  n_duplicates  constant_or_empty
        feature_A    int64          0         3             1              False
        feature_B   object          0         3             1              False
        target_val float64          0         4             0              False
        <BLANKLINE>
        Number of rows with duplicates in input configuration: 1.
        Number of rows with same values: 0.

    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Argument `df` is not pd.DataFrame.")
    if not isinstance(output_cols, list):
        raise TypeError(f"Argument `output_cols` is list.")
    if df.empty:
        raise ValueError(f"DataFrame is empty or contains no data.")

    data_overview = pd.DataFrame({
        'dtypes': df.dtypes,
        'n_missing': df.isna().sum(),
        'n_unique': df.nunique(),
        'n_duplicates': df.apply(
            lambda column: column.duplicated().sum()) # pouzije funkciu okolo osi DataFrame-u.
    })

    data_overview['constant_or_empty'] = data_overview['n_unique'] == 1

    safe_output_cols = output_cols if output_cols is not None else []
    input_like_cols = [col for col in df.columns if col not in safe_output_cols]
    n_duplicate_inputs = df.duplicated(subset=input_like_cols).sum()

    data_text = {
        'df_columns': df.shape[1],
        'df_rows': df.shape[0],
        'n_duplicate_inputs': n_duplicate_inputs,
        'n_identical_rows': df.duplicated(keep=False).sum() # keep=False: spocita vsetky vyskyty hodnoty
    }

    if verbose:
        print(f"DataFrame size: {data_text.get('df_columns')} columns and {data_text.get('df_rows')} rows.")
        print()
        print(data_overview)
        print()
        print(f"Number of rows with duplicates in input configuration: {data_text.get('n_duplicate_inputs')}.")
        print(f"Number of rows with same values: {data_text.get('n_identical_rows')}.")

    if generate_output:
        data_writer = DataWriter()
        data_writer.write_data(data_overview, data_text, filename="overview.json")

    return None