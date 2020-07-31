"""This module contains functions for quickly evaluating the modelling capabilities of a dataset.

The main two functions are:
    - predictive_modelling_score(df, y_label, x_labels, model)
    - predictive_modelling_comparison(df, synth_df, y_label, x_labels, model)

The functions handle Categorical and Continuous values. When the y_label is a continuous value the functions compute
an R^2 value for a regression task; And when the y_label is a categorical value, the functions compute a ROC AUC value
for a binary/multinomial classification task.
"""
from typing import Union, Dict, List, Type, Optional, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

from sklearn.base import ClassifierMixin
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from .dataset import categorical_or_continuous_values
from ..metadata import MetaExtractor, CategoricalMeta, ValueMeta
from ..version import versionadded

#  TODO: Make the keys Enum members.
CLASSIFIERS: Dict[str, Type[ClassifierMixin]] = {
    'Linear': RidgeClassifier,
    'Logistic': LogisticRegression,
    'GradientBoosting': GradientBoostingClassifier,
    'RandomForest': RandomForestClassifier,
    'MLP': MLPClassifier,
    'LinearSVM': LinearSVC
}
"""A dictionary of sklearn classifiers with fit/predict methods."""

REGRESSORS: Dict[str, Type[RegressorMixin]] = {
    'Linear': Ridge,
    'GradientBoosting': GradientBoostingRegressor,
    'RandomForest': RandomForestRegressor,
    'MLP': MLPRegressor,
    'LinearSVM': LinearSVR
}
"""A dictionary of sklearn regressors with fit/predict methods."""

MAX_ANALYSIS_SAMPLE_SIZE = 10_000
RANDOM_SEED = 42


@versionadded('1.0.0')
def predictive_modelling_score(data: pd.DataFrame, y_label: str, x_labels: List[str], model: str):
    """Calculates an R2 or ROC AUC score for a dataset for a given model and labels.

    This function will fit a regressor or classifier depending on the datatype of the y_label. All necessary
    preprocessing (standard scaling, one-hot encoding) is done in the function.

    Args:
        data: The input dataset.
        y_label: The name of the target variable column/response variable.
        x_labels: A list of the input column names/explanatory variables.
        model: One of 'Linear', 'GradientBoosting', 'RandomForrest', 'MLP', 'LinearSVM', or 'Logistic'. Note that
            'Logistic' only applies to categorical response variables.

    Returns:
        The score, metric ('r2' or 'roc_auc'), and the task ('regression', 'binary', or 'multinomial')
    """
    score, metric, task = None, None, None

    available_columns = list(set(data.columns).intersection(set(x_labels+[y_label])))
    if y_label not in available_columns or len([label for label in x_labels if label in available_columns]) == 0:
        raise ValueError('Response variable not in DataFrame.')

    data = data[available_columns]
    dp = MetaExtractor.extract(df=data)

    categorical, continuous = categorical_or_continuous_values(dp)
    available_columns = [v.name for v in cast(List[ValueMeta], categorical)+continuous]

    dp.values = cast(List[ValueMeta], categorical)+continuous
    dp.columns = available_columns
    data = data[available_columns]
    x_labels = list(filter(lambda v: v in available_columns, x_labels))

    if y_label not in available_columns:
        raise ValueError('Response variable type not handled.')
    elif len(x_labels) == 0:
        raise ValueError('No explanatory variables with acceptable type.')
    elif y_label in [v.name for v in categorical] and model not in CLASSIFIERS:
        raise KeyError(f'Selected model ({model}) not available for classification.')
    elif y_label in [v.name for v in continuous] and model not in REGRESSORS:
        raise KeyError(f'Selected model ({model}) not available for regression.')

    sample_size = min(MAX_ANALYSIS_SAMPLE_SIZE, len(data))
    x_train, y_train, x_test, y_test = _preprocess_split_data(data, dp, y_label, x_labels, sample_size)

    if y_label in [v.name for v in continuous]:
        metric = 'r2'
        task = 'regression'
        y_val = [val for val in continuous if val.name == y_label][0]
        score = regressor_score(x_train, y_train, x_test, y_test, model, y_val)

    elif y_label in [v.name for v in categorical]:
        y_val = [val for val in categorical if val.name == y_label][0]
        num_classes = y_val.num_categories
        metric = 'roc_auc' if num_classes == 2 else 'macro roc_auc'
        task = 'binary' if num_classes == 2 else f'multinomial [{num_classes}]'
        score = classifier_score(x_train, y_train, x_test, y_test, model)

    return score, metric, task


@versionadded('1.0.0')
def predictive_modelling_comparison(data: pd.DataFrame, synth_data: pd.DataFrame,
                                    y_label: str, x_labels: List[str], model: str):
    score, synth_score, metric, task = None, None, None, None

    available_columns = list(set(data.columns).intersection(set(x_labels+[y_label])))
    if y_label not in available_columns or len([label for label in x_labels if label in available_columns]) == 0:
        raise ValueError('Response variable not in DataFrame.')

    data = data[available_columns]
    dp = MetaExtractor.extract(df=data)

    categorical, continuous = categorical_or_continuous_values(dp)
    available_columns = [v.name for v in cast(List[ValueMeta], categorical) + continuous]

    dp.values = cast(List[ValueMeta], categorical) + continuous
    dp.columns = available_columns
    data = data[available_columns].dropna()
    synth_data = synth_data[available_columns].dropna()

    x_labels = list(filter(lambda v: v in available_columns, x_labels))

    if y_label not in available_columns:
        raise ValueError('Response variable type not handled.')
    elif len(x_labels) == 0:
        raise ValueError('No explanatory variables with acceptable type.')
    elif y_label in [v.name for v in categorical] and model not in CLASSIFIERS:
        raise KeyError(f'Selected model ({model}) not available for classification.')
    elif y_label in [v.name for v in continuous] and model not in REGRESSORS:
        raise KeyError(f'Selected model ({model}) not available for regression.')

    sample_size = min(MAX_ANALYSIS_SAMPLE_SIZE, len(data), len(synth_data))
    x_train, y_train, x_test, y_test = _preprocess_split_data(data, dp, y_label, x_labels, sample_size)
    x_synth, y_synth = _preprocess_data(synth_data, dp, y_label, x_labels, int(0.8*sample_size))

    if y_label in [v.name for v in continuous]:
        metric = 'r2'
        task = 'regression'
        y_val = [val for val in continuous if val.name == y_label][0]
        score = regressor_score(x_train, y_train, x_test, y_test, model, y_val)
        synth_score = regressor_score(x_synth, y_synth, x_test, y_test, model, y_val)

    elif y_label in [v.name for v in categorical]:
        y_val = [val for val in categorical if val.name == y_label][0]
        num_classes = y_val.num_categories
        metric = 'roc_auc' if num_classes == 2 else 'macro roc_auc'
        task = 'binary' if num_classes == 2 else f'multinomial [{num_classes}]'
        score = classifier_score(x_train, y_train, x_test, y_test, model)
        synth_score = classifier_score(x_synth, y_synth, x_test, y_test, model)

    return score, synth_score, metric, task


def classifier_score(x_train, y_train, x_test, y_test, model) -> float:

    if len(np.unique(y_train)) == 1:
        return 1.

    else:
        clf = CLASSIFIERS[model]()
        clf.fit(x_train, y_train)

        # Two classes classification
        if len(np.unique(y_train)) == 2:
            if hasattr(clf, 'predict_proba'):
                y_pred_test = clf.predict_proba(x_test).T[1]
            else:
                y_pred_test = clf.predict(x_test)

            return roc_auc_score(y_test, y_pred_test)

        # Multi-class classification
        else:
            oh = OneHotEncoder()
            oh.fit(np.concatenate((y_train, y_test)).reshape(-1, 1))
            y_test = oh.transform(y_test.reshape(-1, 1)).toarray()

            if hasattr(clf, 'predict_proba'):
                y_pred_test = clf.predict_proba(x_test)
            else:
                y_pred_test = oh.transform(clf.predict(x_test).reshape(-1, 1)).toarray()

            y_test, y_pred_test = _remove_zero_column(y_test, y_pred_test)
            return roc_auc_score(y_test, y_pred_test, multi_class='ovo')


def regressor_score(x_train, y_train, x_test, y_test, model, y_val) -> float:
    rgr = REGRESSORS[model]()
    rgr.fit(x_train, y_train)

    f_test = rgr.predict(x_test)

    y_test = y_val.postprocess(pd.DataFrame({y_val.name: y_test})).values
    f_test = y_val.postprocess(pd.DataFrame({y_val.name: f_test})).values

    return r2_score(y_test, f_test)


def logistic_regression_r2(df, y_label: str, x_labels: List[str], **kwargs) -> Union[None, float]:
    dp = kwargs.get('dp')
    if dp is None:
        dp = MetaExtractor.extract(df=df)
    categorical, continuous = categorical_or_continuous_values(dp)
    if y_label not in [v.name for v in categorical]:
        return None
    if len(x_labels) == 0:
        return None

    df = dp.preprocess(df)

    df = df[x_labels+[y_label]].dropna()
    df = df.sample(min(MAX_ANALYSIS_SAMPLE_SIZE, len(df)))

    if df[y_label].nunique() < 2:
        return None

    x_array = _preprocess_x2(df[x_labels], None, [v for v in categorical if v.name in x_labels])
    y_array = df[y_label].values

    rg = LogisticRegression()
    rg.fit(x_array, y_array)

    labels = df[y_label].map({c: n for n, c in enumerate(rg.classes_)}).to_numpy()
    oh_labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))

    lp = rg.predict_log_proba(x_array)
    llf = np.sum(oh_labels * lp)

    rg = LogisticRegression()
    rg.fit(np.ones(x_array.shape), y_array)

    lp = rg.predict_log_proba(x_array)
    llnull = np.sum(oh_labels * lp)

    psuedo_r2 = 1 - (llf / llnull)

    return psuedo_r2


def _preprocess_data(data, dp, response_variable, explanatory_variables, sample_size):
    data = dp.preprocess(data)
    sample = data.sample(sample_size, random_state=RANDOM_SEED)

    categorical, continuous = categorical_or_continuous_values(dp)

    df_x, y = sample[explanatory_variables], sample[response_variable].values
    x = _preprocess_x2(df_x, None, [v for v in categorical if v.name in explanatory_variables])

    return x, y


def _preprocess_split_data(data, dp, response_variable, explanatory_variables, sample_size):
    data = dp.preprocess(data)
    sample = data.sample(sample_size, random_state=RANDOM_SEED)

    categorical, continuous = categorical_or_continuous_values(dp)

    if response_variable in [v.name for v in categorical]:
        if all(sample[response_variable].value_counts().values > 1):
            df_train, df_test = train_test_split(sample, test_size=0.2, stratify=sample[response_variable],
                                                 random_state=RANDOM_SEED)
        else:
            df_train, df_test = train_test_split(sample, test_size=0.2, random_state=RANDOM_SEED)

            train_unique = df_train[response_variable].unique()
            target_in_train = df_test[response_variable].apply(lambda y: True if y in train_unique else False)
            df_test = df_test[target_in_train]

    else:
        df_train, df_test = train_test_split(sample, test_size=0.2, random_state=RANDOM_SEED)

    df_x_train, y_train = df_train[explanatory_variables], df_train[response_variable].values
    df_x_test, y_test = df_test[explanatory_variables], df_test[response_variable].values
    x_train, x_test = _preprocess_x2(df_x_train, df_x_test, [v for v in categorical if v.name in explanatory_variables])

    return x_train, y_train, x_test, y_test


def _remove_zero_column(y1, y2):
    """Given two one-hot encodings, delete all columns from both arrays if they are all zeros for any of the two arrays.

    Args:
        y1, y2: Input arrays

    Returns:
        y1, y2: Output arrays

    """
    if len(y1.shape) != 2 or len(y2.shape) != 2:
        return y1, y2
    assert y1.shape[1] == y2.shape[1]

    delete_index = np.where((y1 == 0).all(axis=0) | (y2 == 0).all(axis=0))

    y1 = np.delete(y1, delete_index, axis=1)
    y2 = np.delete(y2, delete_index, axis=1)

    return y1, y2


def _preprocess_x2(x_train: pd.DataFrame, x_test: Optional[pd.DataFrame], values_categorical: List[CategoricalMeta]):
    x_all = pd.concat((x_train, x_test), axis='index') if x_test is not None else x_train
    train_result = []
    test_result = []

    columns_categorical = [v.name for v in values_categorical]
    columns_numeric = [col for col in x_train.columns if col not in columns_categorical]

    if len(columns_numeric) > 0:
        train_result.append(x_train[columns_numeric])
        if x_test is not None:
            test_result.append(x_test[columns_numeric])

    if len(columns_categorical) > 0:
        oh = OneHotEncoder(categories=[list(range(v.num_categories or 0)) for v in values_categorical], sparse=False)
        oh.fit(x_all[columns_categorical])
        train_result.append(oh.transform(x_train[columns_categorical]))
        if x_test is not None:
            test_result.append(oh.transform(x_test[columns_categorical]))

    if x_test is not None:
        return np.concatenate(train_result, axis=1), np.concatenate(test_result, axis=1)
    else:
        return np.concatenate(train_result, axis=1)
