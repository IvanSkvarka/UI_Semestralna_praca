from pathlib import Path
from typing import Union
import json
import pandas as pd
import matplotlib.pyplot as plt

from data_handling.data_handling import DataHandler

class DataWriter(DataHandler):
    """
    Ukladá výsledky do priečinka reports.
    Podľa typu dát sám vyberie formát:
        pd.DataFrame  -> .csv
        dict          -> .json
        plt.Figure    -> .png | .pdf
        str           -> .txt
    """

    def __init__(self, output_dir: Union[str, Path] = "reports"):
        """
        output_dir - priečinok, kam sa bude ukladať (napr. "reports")
        """
        super().__init__(output_dir)


    def _set_graph_export(self) -> None:
        """Set graph to TeX style."""
        plt.rcParams.update({
            'pgf.preamble': "\n".join([
                r"\usepackage{mathptmx}",
            ]),
            'font.family': "serif",
            'font.serif': ["Times", "Times New Roman", "ptm"],
            'font.size': 12,
            'xtick.labelsize': 12 * 0.8,
            'ytick.labelsize': 12 * 0.8,
            'legend.fontsize': 12 * 0.6,
            'pgf.rcfonts': False,
            'text.usetex': True,
            'pgf.texsystem': "pdflatex",
        })

        return None

    def _reset_graph_export(self) -> None:
        """Reset graph style to default."""
        plt.rcParams.update(plt.rcParamsDefault)

        return None


    def write_data(self, data, *args, filename: str = "output", subfolder: str = "") -> Path:
        """
        data      - čo chceme uložiť (DataFrame, dict, Figure alebo str)
        filename  - názov súboru BEZ prípony (prípona sa doplní sama)
        subfolder - podpriečinok v reports, napr. "figures" (nepovinné)
        Vráti cestu k uloženému súboru.
        """
        # 1) Poskladanie priečinka a jeho vytvorenie, ak ešte neexistuje
        folder = self.folder / subfolder
        folder.mkdir(parents=True, exist_ok=True)

        # 2) Podľa typu dát vyberieme príponu
        if isinstance(data, pd.DataFrame):
            extension = "csv"
        elif isinstance(data, dict):
            extension = "json"
        elif isinstance(data, plt.Figure):
            extension = "png"
            if args and args[0] == True:
                extension = "pdf"
        elif isinstance(data, str):
            extension = "txt"
        else:
            raise ValueError(f"Data type '{type(data).__name__}' cannot be saved.")

        filepath = folder / f"{filename}.{extension}"

        # 3) Samotné uloženie
        try:
            if extension == "csv":
                data.to_csv(filepath)

            elif extension == "json":
                with open(filepath, "w", encoding="utf-8") as file:
                    # default=str: numpy čísla a iné typy prevedie na text
                    json.dump(data, file, indent=4, ensure_ascii=False, default=str)

            elif extension == "png" or extension == "pdf":
                if extension == "png":
                    data.savefig(filepath, dpi=120, bbox_inches="tight", backend="Agg")
                if extension == "pdf":
                    self._set_graph_export()
                    data.savefig(filepath, dpi=120, bbox_inches="tight", backend="pgf")
                    self._reset_graph_export()
                plt.close(data)          # zavrie graf, aby nezaberal pamäť

            elif extension == "txt":
                with open(filepath, "w", encoding="utf-8") as file:
                    file.write(data)

        except Exception as e:
            raise ValueError(f"Error saving file '{filepath}': {e}.")

        print(f"Saved: {filepath}")
        return filepath
