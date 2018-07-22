import warnings
warnings.filterwarnings(action='ignore', message="numpy.dtype size changed")

import unittest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from synthesized.core import BasicSynthesizer
from synthesized.testing.testing_environment import estimate_utility


class TestEncodings(unittest.TestCase):

    def test_basic(self):
        data = pd.read_csv('data/transactions.csv')
        # data = data.head(1000)

        data = data[['type', 'operation', 'amount']]
        data = data.dropna()
        data = data[data['type'] != 'VYBER']
        data['type'] = data['type'].astype(dtype='int')
        data['type'] = data['type'].astype(dtype='category')
        data['operation'] = data['operation'].astype(dtype='int')
        data['operation'] = data['operation'].astype(dtype='category')
        data['amount'] = data['amount'].astype(dtype='float32')

        with BasicSynthesizer(dtypes=data.dtypes) as synthesizer:
            synthesizer.learn(data=data, verbose=5000)
            synthesized = synthesizer.synthesize(n=10000)

        print(data['type'].value_counts(normalize=True, sort=False))
        print(synthesized['type'].value_counts(normalize=True, sort=False))
        print(data['operation'].value_counts(normalize=True, sort=False))
        print(synthesized['operation'].value_counts(normalize=True, sort=False))
        bins = [
            min(data['amount'].min(), 0e3), 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,
            max(data['amount'].max(), 10e3)
        ]
        print(pd.cut(data['amount'], bins=bins, include_lowest=True).value_counts(
            normalize=True, sort=False
        ))
        bins = [
            min(synthesized['amount'].min(), 0e3), 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,
            max(synthesized['amount'].max(), 10e3)
        ]
        print(pd.cut(synthesized['amount'], bins=bins, include_lowest=True).value_counts(
            normalize=True, sort=False
        ))

        print(estimate_utility(
            df_orig=data, df_synth=synthesized,
            continuous_columns=['amount'], categorical_columns=['type', 'operation'],
            classifier=DecisionTreeClassifier(), regressor=DecisionTreeRegressor()
        ))
