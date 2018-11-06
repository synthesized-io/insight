import pandas as pd
from pathlib import Path

path = Path(__file__).parent.parent / 'data/credit.csv'


def load_data():
    data = pd.read_csv(path)
    data.drop('Unnamed: 0', axis=1, inplace=True)
    data.dropna(inplace=True)
    data['SeriousDlqin2yrs'] = data['SeriousDlqin2yrs'].astype(dtype='category')
    return data
