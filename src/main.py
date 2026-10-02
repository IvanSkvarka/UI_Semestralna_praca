from data_handling import DataLoader
import pandas as pd

if __name__ == "__main__":
    data_loader = DataLoader("data/raw")
    data = data_loader.load_data("training_data_set")
