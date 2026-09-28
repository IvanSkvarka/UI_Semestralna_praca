import pandas as pd

def overview(df: pd.DataFrame, generate_output: bool = True, verbose: bool = True) -> None:
    """

    """

    data_overview = pd.DataFrame({
        'dtypes': df.dtypes,
        'n_missing': df.isna().sum(),
        'n_unique': df.nunique(),
        'n_duplicates': df.apply(lambda column: column.duplicated().sum()), # Apply - pouzije fukciu okolo osi DataFrame-u.
    })

    data_overview['constant_or_empty'] = data_overview['n_unique'] == 1

    input_like_cols = [col for col in df.columns if col not in
                        ['strainRate_p50', 'epsilon_p50', 'k_p50', 'Umag_p50']]
    n_duplicate_inputs = df.duplicated(subset=input_like_cols).sum()

    data_text = {
        'df_columns': df.shape[1],
        'df_rows': df.shape[0],
        'n_duplicate_inputs': n_duplicate_inputs,
        'n_identical_rows': df.duplicated(keep=False).sum() # keep=False: spocita vsetky vyskyty hodnoty
    }

    if verbose:
        print(f"DataFrame size: {df.shape[1]} columns and {df.shape[0]} rows.")
        print()
        print(data_overview)
        print()
        print(f"Number of rows with duplicates in input configuration: {data_text.get('n_duplicate_inputs')}.")
        print(f"Number of rows with same values: {data_text.get('n_identical_rows')}.")

    if generate_output:
        pass

    return None
