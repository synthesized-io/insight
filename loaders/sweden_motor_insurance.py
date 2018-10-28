import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

path = Path(__file__).parent.parent / 'data/insurance/sweden_motor_insurance/sweden_motor_insurance.csv'


def load_data(test_size=0.2):
    data = pd.read_csv(path)
    return train_test_split(data, test_size=test_size, random_state=0)
