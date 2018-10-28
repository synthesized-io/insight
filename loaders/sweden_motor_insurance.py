import pandas as pd
from pathlib import Path

path = Path(__file__).parent.parent / 'data/insurance/sweden_motor_insurance/sweden_motor_insurance.csv'


def load_data():
    return pd.read_csv(path)

