import pandas as pd
from pathlib import Path

path = Path(__file__).parent.parent / 'data/james_with_categoricals.csv'


def load_data():
    data = pd.read_csv(path)
    data.dropna(inplace=True)
    data['SeriousDlqin2yrs'] = data['SeriousDlqin2yrs'].astype(dtype='category')
    return data
