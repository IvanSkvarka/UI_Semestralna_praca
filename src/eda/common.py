import pandas as pd

def get_constant_columns(df: pd.DataFrame) -> list:
    """Selects columns that are constant.
    Appends column into result, if there is only one unique value.
    
    Args:
        df (pd.DataFrame): DataFrame containing dataset. 
    Returns:
        list: A list of constant columns.
    """
    result = []
    for column in df.columns:
        if df[column].nunique() == 1:
            result.append(column)
    return result

def get_input_columns(df: pd.DataFrame, output_columns: list = None) -> list:
    """Selects input columns.
    
    Appends input column into `input_columns`, if it is specified.
    Needed because output columns in traing set leads to data leakage.
    
    Args:
        df (pd.DataFrame): DataFrame containing dataset. 
        output_columns (list): list of output_columns.
    Returns:
        list: A list of input columns.
    """
    # TODO: dokoncit docstring
    if output_columns is None:
        output_columns = []
    input_columns = []
    for column in df.columns:
        if column not in output_columns:
            input_columns.append(column)
    return input_columns

def get_numeric_columns(df: pd.DataFrame, columns: list) -> list:  
    """Selects numeric columns.
    
    Appends numeric column into `numeric_columns`, if it is speciafied.
    Needed because output columns in traing set leads to data leakage.
    
    Args:
        df (pd.DataFrame): DataFrame containing dataset. 
        output_columns (list): list of output_columns.
    Returns:
        list: A list of input columns.
    """
    # TODO: dokoncit docstring
    numeric_columns = []  
    for column in columns:  
        if pd.api.types.is_numeric_dtype(df[column]):  
            numeric_columns.append(column)  
    return numeric_columns

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data from constant columns.
    
    Drops constant columns by df.drop and resets index, that means
    creating new index by dropping old one.
    
    Args:
        df (pd.DataFrame): DataFrame 
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # TODO: dokoncit docstring
    cleaned = df.drop(columns=get_constant_columns(df))
    cleaned = cleaned.reset_index(drop=True) # Reset index: creates new index, old is dropped
    return cleaned