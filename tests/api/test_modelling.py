import logging

import pandas as pd

from synthesized.api.modelling import predictive_modelling_comparison, predictive_modelling_score

logger = logging.getLogger(__name__)


def test_modelling_score():
    df = pd.read_csv('data/credit_small.csv')
    score, metric, task = predictive_modelling_score(
        data=df.sample(1000), y_label='age', x_labels=['MonthlyIncome', 'NumberOfTimes90DaysLate'], model='Linear'
    )
    assert score is not None
    assert metric == 'r2_score'
    assert task == 'regression'


def test_modelling_comparison():
    df = pd.read_csv('data/credit_small.csv')
    score, score2, metric, task = predictive_modelling_comparison(
        data=df.sample(1000), synth_data=df.sample(1000), y_label='age',
        x_labels=['MonthlyIncome', 'NumberOfTimes90DaysLate'], model='Linear'
    )

    assert score is not None
    assert score2 is not None
    assert metric == 'r2_score'
    assert task == 'regression'
