import pandas as pd


def load_data() -> pd.DataFrame:
    return pd.read_csv("data/data.csv", sep=";")
