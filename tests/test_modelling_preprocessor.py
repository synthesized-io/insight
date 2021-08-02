import pandas as pd

from src.synthesized_insight.modelling import ModellingPreprocessor


def test_modelling_preprocessor():
    data = pd.read_csv('datasets/credit_with_categoricals.csv')
    p = ModellingPreprocessor(target='SeriousDlqin2yrs')
    p.fit_transform(data)
