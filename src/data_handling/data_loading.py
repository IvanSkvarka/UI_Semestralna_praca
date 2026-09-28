from pathlib import Path
from typing import Union
import pandas as pd

class DataLoader:
    """
    
    """

    def __init__(self, data_dir: Union[str, Path] = "data"):
        """
        Initializes the DataLoader.
        Args:
            data_dir (str): The directory key where data files are located.
        """
        self.data_load_path = Path(data_dir)

    def load_data(self, filename: str = "data") -> pd.DataFrame:
        """
        Load data from a specified CSV file.

        Args:
            filename (str): The name of the CSV file to load (without extension).
        Returns:
            pd.DataFrame
        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            ValueError: If there are issues loading the CSV or empty data.
        """
        filepath = self.data_load_path / f"{filename}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"File '{filepath}' not found")

        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Error loading CSV file '{filepath}': {e}")

        if data.empty:
            raise ValueError(f"File '{filepath}' is empty or contains no data")

        return data