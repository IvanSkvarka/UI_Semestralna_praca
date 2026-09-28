from pathlib import Path
from typing import Union, Any
import pandas as pd
import json
import numpy as np

class PandasJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy and Pandas data types.

    This encoder intercepts non-native Python data types during the JSON 
    dump process and converts them into standard, JSON-compatible formats 
    (e.g., NumPy integers to Python ints, Pandas dtypes to strings).
    """

    def default(self, obj: Any) -> Any:
        """Convert a non-serializable object into a native Python type.

        Args:
            obj (Any): The object to inspect and potentially convert.

        Returns:
            Any: A JSON-serializable representation of the object.
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        
        elif hasattr(obj, 'name') or isinstance(obj, (np.dtype, type)):
            return str(obj)

        return super().default(obj)

class DataWriter:
    """Initialize the DataWriter with a target destination directory.

    Args:
        data_dir (Union[str, Path], optional): The destination directory path 
            where files will be created. Defaults to "reports".
    """

    def __init__(self, data_dir: Union[str, Path] = "reports"):
        """

        """
        self.data_write_path = Path(data_dir)

    def write_data(self, *args: Any, filename: str = "data.json", **kwargs: Any) -> None:
        """Export one or multiple data structures into a single JSON or CSV file.

        All positional arguments (`*args`) are captured exclusively as data payloads.
        For JSON exports, complex structures like pandas DataFrames and custom numpy 
        types (`int64`, `float64`) are automatically handled via `PandasJSONEncoder`.


        Args:
            *args (Any): Variable length argument list where every single element 
                represents a data payload (e.g., DataFrames, dicts, lists) to write.
            filename (str, optional): The target filename as a keyword argument. 
                Must end with either '.json' or '.csv'. Defaults to "data.json".
            **kwargs (Any): Arbitrary keyword arguments forwarded directly to the 
                underlying serialization mechanism (e.g., `index=False` for pandas CSV, 
                or `indent=4` for JSON blocks).

        Returns:
            None: This method writes data directly to the disk subsystem.

        Raises:
            ValueError: If no data arguments are passed via `*args`, or if the 
                `filename` extension is unsupported.
            TypeError: If the data structures provided in `*args` are incompatible 
                with the requested CSV format.
            OSError: If the directory cannot be created or writing fails due to 
                OS permissions.

        Examples:
            >>> import pandas as pd
            >>> writer = DataWriter(data_dir="my_reports")
            
            >>> # 1. Writing a dictionary to JSON (filename must be a keyword)
            >>> config = {"mode": "train", "lr": 0.01}
            >>> writer.write_data(config, filename="config.json")
            
            >>> # 2. Writing MULTIPLE items to JSON (they get grouped into a list)
            >>> p1 = {"name": "Alice"}
            >>> p2 = {"name": "Bob"}
            >>> writer.write_data(p1, p2, filename="users.json")
            
            >>> # 3. Writing MULTIPLE DataFrames to CSV (they get concatenated)
            >>> df1 = pd.DataFrame({'val': [1, 2]})
            >>> df2 = pd.DataFrame({'val': [3, 4]})
            >>> writer.write_data(df1, df2, filename="combined.csv", index=False)
        """
        if not args:
            raise ValueError("No data payload was provided. Please pass your dataset as a positional argument.")

        self.data_write_path.mkdir(parents=True, exist_ok=True) # Vytvorit adresar ak neexistuje
        full_path = self.data_write_path / filename
        file_suffix = full_path.suffix.lower() # Rozhodovanie podla sufixu

        if file_suffix not in ['.json', '.csv']:
            raise ValueError(
                f"Unsupported file extension '{file_suffix}'. Supported formats: '.json' and '.csv' ."
            )

        if file_suffix == '.json':
            processed_args = []
            for item in args:
                if isinstance(item, pd.DataFrame):
                    # Převod DataFrame na slovník s orientací na index (vhodné pro data_overview)
                    processed_args.append(item.to_dict(orient='index'))
                else:
                    processed_args.append(item)

            final_data = processed_args if len(processed_args) == 1 else processed_args

            if 'indent' not in kwargs:
                kwargs['indent'] = 4
            kwargs['cls'] = PandasJSONEncoder
            
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, **kwargs)

        elif file_suffix == '.csv':
            if not all(isinstance(item, pd.DataFrame) for item in args):
                raise TypeError("CSV export is exclusively supported for pandas DataFrame structures.")
            
            final_df = args if len(args) == 1 else pd.concat(args, ignore_index=True)
            final_df.to_csv(full_path, **kwargs)

        return None