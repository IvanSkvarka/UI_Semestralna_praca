from data_handling import DataLoader, DataWriter
import pandas as pd
from eda import overview, relationships, design

if __name__ == "__main__":
    data_loader = DataLoader("data/raw")
    data_writer = DataWriter("reports")
    data = data_loader.load_data("training_data_set")

    output_cols = ['strainRate_p50', 'epsilon_p50', 'k_p50', 'Umag_p50']
    target = 'epsilon_p50'

    overview(data, output_cols, 2, data_writer=data_writer)
    relationships(data, target, output_cols, data_writer=data_writer)
    design(data, target, output_cols, 'rpm')