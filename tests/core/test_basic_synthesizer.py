import unittest
import pandas as pd
from synthesized.core import BasicSynthesizer


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
        synthesizer = BasicSynthesizer(dtypes=data.dtypes)
        synthesizer.initialize()
        synthesizer.learn(data=data, verbose=True)
        synthesized = synthesizer.synthesize(n=10000)
        print(data['type'].value_counts(normalize=True, sort=False))
        print(synthesized['type'].value_counts(normalize=True, sort=False))
        print(data['operation'].value_counts(normalize=True, sort=False))
        print(synthesized['operation'].value_counts(normalize=True, sort=False))
        bins = [
            min(data['amount'].min(), 0e3), 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,
            max(data['amount'].max(), 10e3)
        ]
        print(pd.cut(data['amount'], bins=bins).value_counts(normalize=True, sort=False))
        bins = [
            min(synthesized['amount'].min(), 0e3), 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,
            max(synthesized['amount'].max(), 10e3)
        ]
        print(pd.cut(synthesized['amount'], bins=bins).value_counts(normalize=True, sort=False))
