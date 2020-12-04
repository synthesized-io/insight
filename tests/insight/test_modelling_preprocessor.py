import numpy as np
import pandas as pd
import pytest

from synthesized.insight.modelling import ModellingPreprocessor
from synthesized.insight.metrics import predictive_modelling_score


@pytest.mark.fast
def test_modelling_preprocessor():
    data = pd.read_csv('data/credit_with_categoricals.csv')
    p = ModellingPreprocessor(target='SeriousDlqin2yrs')
    p.fit_transform(data)


@pytest.mark.fast
def test_predictive_modelling_score_clf():
    n = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.where(np.random.uniform(size=n) < 0.8, np.random.randn(n), np.nan),
        'x3': np.random.choice(['a', 'b', 'c', 'd'], size=n),
        'x4': np.where(np.random.uniform(size=n) < 0.8, np.random.randint(1e5, size=n).astype(str), ''),
        'y': np.random.choice([0, 1], size=n)
    })

    target = 'y'
    x_labels = list(filter(lambda c: c != target, data.columns))
    predictive_modelling_score(data, model='Logistic', y_label=target, x_labels=x_labels)


@pytest.mark.fast
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
    predictive_modelling_score(data, model='Linear', y_label=target, x_labels=x_labels)
