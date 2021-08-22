
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

from src.synthesized_insight.modelling import ModellingPreprocessor
from src.synthesized_insight.modelling.utility import check_model_type, sample_split_data


@pytest.fixture(scope='module')
def df():
    df = pd.read_csv('datasets/credit_with_categoricals.csv')
    return df


def test_sample_split_data(df):
    response_variable = 'NumberOfDependents'
    df_train, df_test = sample_split_data(df, response_variable=response_variable)
    assert df_train is not None and df_test is not None
    train_unique_targets = df_train[response_variable].unique()
    test_unique_targets = df_test[response_variable].unique()
    assert all([test_target in train_unique_targets for test_target in test_unique_targets])


def test_modelling_preprocessor(df):
    data = pd.read_csv('datasets/credit_with_categoricals.csv')
    prep = ModellingPreprocessor(target='SeriousDlqin2yrs')
    assert prep.fit_transform(data) is not None


def test_check_model_type():
    assert isinstance(check_model_type('RandomForest', False, 'clf'), RandomForestClassifier)

    with pytest.raises(KeyError):
        check_model_type('AdaBoost', False, 'clf')

    with pytest.raises(ValueError):
        check_model_type(None, False, 'clf')

    assert isinstance(check_model_type('GradientBoosting', False, 'rgr'), GradientBoostingRegressor)

    with pytest.raises(KeyError):
        check_model_type('AdaBoost', False, 'rgr')

    with pytest.raises(ValueError):
        check_model_type(None, False, 'rgr')
