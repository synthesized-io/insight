import random
import string

import pandas as pd
from faker import Faker

from synthesized.privacy import DataMasker


def test_data_masker():
    data = pd.read_csv('data/credit_with_categoricals.csv')
    fkr = Faker()
    data['Name'] = [fkr.name() for _ in range(len(data))]
    data['ID'] = [''.join([random.choice(string.hexdigits) for _ in range(16)]) for _ in range(len(data))]

    masked_columns = dict(age='rounding|3', MonthlyIncome='rounding',
                          effort='partial_masking|0.25', ID='partial_masking',
                          NumberOfOpenCreditLinesAndLoans='random|8', Name='random',
                          SeriousDlqin2yrs='null')

    data_masker = DataMasker(masked_columns)
    data_masked = data_masker.fit_transform(data)

    assert data.shape == data_masked.shape
