from pathlib import Path
from typing import Union

class DataHandler:  
    def __init__(self, folder: Union[str, Path]):  
        self.folder = Path(folder) 