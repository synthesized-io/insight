import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

path = Path(__file__).parent.parent / 'data/credit.csv'


def load_data(test_size=0.2):
    data = pd.read_csv(path)
    data.drop('Unnamed: 0', axis=1, inplace=True)
    data.dropna(inplace=True)
    data['SeriousDlqin2yrs'] = data['SeriousDlqin2yrs'].astype(dtype='category')
    return train_test_split(data, test_size=test_size, random_state=0)
