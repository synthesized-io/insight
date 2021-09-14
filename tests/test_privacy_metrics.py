import copy

import numpy as np
import pandas as pd
import pytest

from synthesized_insight.metrics import privacy_cap_scorer
from synthesized_insight.metrics.privacy import AttributeInferenceAttackCAP, AttributeInferenceAttackML


@pytest.fixture(scope='module')
def df():
    df = pd.read_csv('tests/datasets/mini_credit.csv')
    return df


def check_output(metrics_cls, model, predictors, sensitive_col, df):
    privacy_metrics = metrics_cls(copy.deepcopy(model), sensitive_col, predictors)
    privacy_score_same = privacy_metrics(df, df)

    privacy_metrics = metrics_cls(copy.deepcopy(model), sensitive_col, predictors)
    new_df = df.copy()
    new_df[sensitive_col] = np.random.permutation(new_df[sensitive_col].values)
    privacy_score_permuted = privacy_metrics(new_df, df)
    assert privacy_score_same < privacy_score_permuted


@pytest.mark.slow
@pytest.mark.parametrize(
    "predictors,sensitive_col",
    [
        pytest.param(['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio'], 'age'),
        pytest.param(['age', 'NumberOfDependents', 'SeriousDlqin2yrs'], 'MonthlyIncome'),
        pytest.param(None, 'MonthlyIncome')
    ])
def test_privacy_metrics_numerical(df, predictors, sensitive_col):
    np.random.seed(6235901)
    model = 'Linear'
    check_output(AttributeInferenceAttackML, model, predictors, sensitive_col, df.head(1000).copy())


@pytest.mark.slow
@pytest.mark.parametrize(
    "predictors,sensitive_col",
    [
        pytest.param(['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'age'], 'NumberOfDependents'),
        pytest.param(['NumberOfDependents', 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'age'], 'SeriousDlqin2yrs'),
        pytest.param(None, 'SeriousDlqin2yrs')
    ])
def test_privacy_metrics_categorical_ml(df, predictors, sensitive_col):
    np.random.seed(6235901)
    model = 'RandomForest'
    check_output(AttributeInferenceAttackML, model, predictors, sensitive_col, df.head(1000).copy())


@pytest.mark.parametrize(
    "privacy_cap_model,predictors,sensitive_col",
    [
        pytest.param('GeneralizedCAP', ['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'age'], 'NumberOfDependents'),
        pytest.param('DistanceCAP', ['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'age'], 'NumberOfDependents'),
        pytest.param('GeneralizedCAP', ['SeriousDlqin2yrs', 'age'], 'NumberOfDependents'),
        pytest.param('DistanceCAP', ['SeriousDlqin2yrs', 'age'], 'NumberOfDependents'),
        pytest.param('GeneralizedCAP', None, 'NumberOfDependents')
    ])
def test_privacy_metrics_categorical_cap(df, privacy_cap_model, predictors, sensitive_col):
    np.random.seed(6235901)
    check_output(AttributeInferenceAttackCAP, privacy_cap_model, predictors, sensitive_col, df.head(1000).copy())


def test_privacy_score_ml_categorical(df):
    df = df.head(100)
    predictors = ['NumberOfDependents', 'age']
    sensitive_col = 'SeriousDlqin2yrs'
    ml_eval = AttributeInferenceAttackML('RandomForest', sensitive_col, predictors)
    assert ml_eval._privacy_score_categorical(df[sensitive_col], df[sensitive_col]) == 0

    permuted_sensitive_col = np.random.permutation(df[sensitive_col].values)
    assert ml_eval._privacy_score_categorical(df[sensitive_col], permuted_sensitive_col) > 0


def test_privacy_score_ml_numerical(df):
    df = df.head(100)
    predictors = ['MonthlyIncome', 'NumberOfDependents']
    sensitive_col = 'RevolvingUtilizationOfUnsecuredLines'
    ml_eval = AttributeInferenceAttackML('Linear', sensitive_col, predictors)
    assert ml_eval._privacy_score_numerical(df[sensitive_col], df[sensitive_col]) == 0

    permuted_sensitive_col = np.random.permutation(df[sensitive_col].values)
    assert ml_eval._privacy_score_numerical(df[sensitive_col], permuted_sensitive_col) > 0


def test_get_frequency():
    nums = [1, 1, 2, 3, 1, 2, 2, 1]
    elem = 1
    assert privacy_cap_scorer.get_frequency(nums, elem) == 0.5


def test_get_most_frequent_element():
    nums = [1, 1, 2, 2, 1]
    assert privacy_cap_scorer.get_most_frequent_element(nums) == 1


def test_get_hamming_distance():
    tpl1 = (1, 2, 3)
    assert privacy_cap_scorer.get_hamming_dist(tpl1, tpl1) == 0

    tpl2 = (1, 2, 4)
    assert privacy_cap_scorer.get_hamming_dist(tpl1, tpl2) == 1

    tpl2 = (5, 5, 5)
    assert privacy_cap_scorer.get_hamming_dist(tpl1, tpl2) == 3


def test_get_closest_neighbours():
    tpl_list = [(1, 2, 3), (2, 3, 4), (0, 1, 3), (0, 2, 4), (3, 2, 1)]
    target = (0, 2, 3)

    expected_closest_neighbours = [(1, 2, 3), (0, 1, 3), (0, 2, 4)]
    assert privacy_cap_scorer.get_closest_neighbours(tpl_list, target) == expected_closest_neighbours
