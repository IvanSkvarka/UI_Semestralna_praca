from pathlib import Path
from typing import Union
import pandas as pd

from data_handling.data_handling import DataHandler

class DataLoader(DataHandler):
    """

    """

    def __init__(self, data_dir: Union[str, Path] = "data"):
        """

        """
        super().__init__(data_dir)


    def load_data(self, filename: str = "data") -> pd.DataFrame:
        """

        """
        filepath = self.folder / f"{filename}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"File '{filepath}' not found.")

        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Error loading CSV file '{filepath}': {e}.")

        if data.empty:
            raise ValueError(f"File '{filepath}' is empty or contains no data.")

        return data