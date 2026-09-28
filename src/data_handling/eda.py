import pandas as pd
import data_handling.data_writing as data_writing

def overview(df: pd.DataFrame, generate_output_file: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame):
        generate_output_file (bool):
        verbose (bool):
    Returns:
        pd.DataFrame
    Raises:
    """

    data_overview = pd.DataFrame({
        "dtypes": df.dtypes,
        "n_missing": df.isna().sum(),
        "n_unique": df.nunique().sort_values(),
        "n_duplicates": df.duplicated().sum()

    })

    data_overview["constant_or_empty"] = data_overview["n_unique"] <= 1


    input_like_cols = [column for column in df.columns if column not in
                        ['strainRate_p50', 'epsilon_p50', 'k_p50', 'Umag_p50']]
    n_duplicate_inputs = df.duplicated(subset=input_like_cols).sum()

    data_text = {
        "df_columns": df.shape[1],
        "df_rows": df.shape[0],
        "n_duplicate_inputs": n_duplicate_inputs,
    }

    if verbose:
        print(f"DataFrame size: {df.shape[1]} columns and {df.shape[0]} rows.")
        print(data_overview)
        print(f"Number of columns with duplicates in input configuration: {data_text.get("n_duplicate_inputs")}")

    return data_overview

