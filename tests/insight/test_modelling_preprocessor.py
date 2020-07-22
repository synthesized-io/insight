import numpy as np
import pandas as pd

from synthesized.insight.modelling import ModellingPreprocessor, predictive_modelling_score


def test_modelling_preprocessor():
    data = pd.read_csv('data/credit_with_categoricals.csv')
    p = ModellingPreprocessor(target='SeriousDlqin2yrs')
    p.fit_transform(data)


def test_predictive_modelling_score():
    n = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.where(np.random.uniform(size=n) < 0.8, np.random.randn(n), np.nan),
        'x3': np.random.choice(['a', 'b', 'c', 'd'], size=n),
        'y': np.random.choice([0, 1], size=n)
    })

    target = 'y'
    x_labels = list(filter(lambda c: c != target, data.columns))
    predictive_modelling_score(data, target, x_labels, 'Logistic')


def test_predictive_modelling_score_rgr():
    n = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.where(np.random.uniform(size=n) < 0.8, np.random.randn(n), np.nan),
        'x3': np.random.choice(['a', 'b', 'c', 'd'], size=n),
        'y': np.random.choice([0, 1], size=n)
    })

    target = 'y'
    x_labels = list(filter(lambda c: c != target, data.columns))
    predictive_modelling_score(data, target, x_labels, 'Linear')

