"""This module contains functions for quickly evaluating the modelling capabilities of a dataset.

The main two functions are:
    - predictive_modelling_score(df, y_label, x_labels, model)
    - predictive_modelling_comparison(df, synth_df, y_label, x_labels, model)

The functions handle Categorical and Continuous values. When the y_label is a continuous value the functions compute
an R^2 value for a regression task; And when the y_label is a categorical value, the functions compute a ROC AUC value
for a binary/multinomial classification task.
"""
import logging
from math import sqrt
from typing import Union, Dict, List, Type, Optional, cast, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

from .preprocessor import ModellingPreprocessor
from .metrics import regressor_scores, classifier_scores
from ...metadata import DataFrameMeta, MetaExtractor, ValueMeta

logger = logging.getLogger(__name__)

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


# Used in synthesized metrics
# Used in synthesized-web
def predictive_modelling_score(data: pd.DataFrame, y_label: str, x_labels: Optional[List[str]],
                               model: Union[str, BaseEstimator], copy_model: bool = True,
                               preprocessor: ModellingPreprocessor = None, dp: DataFrameMeta = None):

    """Calculates an R2 or ROC AUC score for a dataset for a given model and labels.

    This function will fit a regressor or classifier depending on the datatype of the y_label. All necessary
    preprocessing (standard scaling, one-hot encoding) is done in the function.

    Args:
        data: The input dataset.
        y_label: The name of the target variable column/response variable.
        x_labels: A list of the input column names/explanatory variables. If none, all except y_label will be used.
        model: One of 'Linear', 'GradientBoosting', 'RandomForrest', 'MLP', 'LinearSVM', or 'Logistic'. Note that
            'Logistic' only applies to categorical response variables.
        copy_model:
        preprocessor:
        dp:

    Returns:
        The score, metric ('r2' or 'roc_auc'), and the task ('regression', 'binary', or 'multinomial')
    """

    score, metric, task = None, None, None

    if x_labels is None:
        x_labels = list(filter(lambda c: c != y_label, data.columns))
    else:
        available_columns = list(set(data.columns).intersection(set(x_labels + [y_label])))
        if y_label not in available_columns or len([label for label in x_labels if label in available_columns]) == 0:
            raise ValueError('Response variable not in DataFrame.')
        data = data[available_columns]

    if dp is None:
        dp = MetaExtractor.extract(df=data)

    categorical, continuous = dp.get_categorical_and_continuous()
    available_columns = [v.name for v in cast(List[ValueMeta], categorical) + continuous]
    data = data[available_columns]
    x_labels = list(filter(lambda v: v in available_columns, x_labels))

    if y_label not in available_columns:
        raise ValueError('Response variable type not handled.')
    elif len(x_labels) == 0:
        raise ValueError('No explanatory variables with acceptable type.')

    # Check predictor
    if y_label in [v.name for v in categorical]:
        estimator = _check_model_type(model, copy_model, 'clf')
    elif y_label in [v.name for v in continuous]:
        estimator = _check_model_type(model, copy_model, 'rgr')
    else:
        raise ValueError(f"Can't understand y_label '{y_label}' type.")

    sample_size = min(MAX_ANALYSIS_SAMPLE_SIZE, len(data))

    if preprocessor is None:
        preprocessor = ModellingPreprocessor(target=y_label, dp=dp)

    df_train_pre, df_test_pre = preprocess_split_data(data, response_variable=y_label, explanatory_variables=x_labels,
                                                      sample_size=sample_size, preprocessor=preprocessor)

    x_labels_pre = list(filter(lambda v: v != y_label, df_train_pre.columns))
    x_train = df_train_pre[x_labels_pre].to_numpy()
    y_train = df_train_pre[y_label].to_numpy()
    x_test = df_test_pre[x_labels_pre].to_numpy()
    y_test = df_test_pre[y_label].to_numpy()

    if y_label in [v.name for v in continuous]:
        metric = 'r2_score'
        task = 'regression'
        scores = regressor_scores(x_train, y_train, x_test, y_test, rgr=estimator, metrics=metric)
        score = scores[metric]

    elif y_label in [v.name for v in categorical]:
        y_val = [val for val in categorical if val.name == y_label][0]
        num_classes = y_val.num_categories
        metric = 'roc_auc' if num_classes == 2 else 'macro roc_auc'
        task = 'binary ' if num_classes == 2 else f'multinomial [{num_classes}]'
        scores = classifier_scores(x_train, y_train, x_test, y_test, clf=estimator, metrics='roc_auc')
        score = scores['roc_auc']

    return score, metric, task


# Used in synthesized metrics
# Used in synthesized-web
def predictive_modelling_comparison(data: pd.DataFrame, synth_data: pd.DataFrame,
                                    y_label: str, x_labels: List[str], model: str):
    score, metric, task = predictive_modelling_score(data, y_label, x_labels, model)
    synth_score, _, _ = predictive_modelling_score(synth_data, y_label, x_labels, model)

    return score, synth_score, metric, task


def preprocess_split_data(data: pd.DataFrame, response_variable: str, explanatory_variables: Optional[List[str]] = None,
                          test_size: float = 0.2, sample_size: Optional[int] = None,
                          preprocessor: ModellingPreprocessor = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = sample_split_data(data, response_variable, explanatory_variables, test_size=test_size,
                                          sample_size=sample_size)

    if preprocessor is None:
        preprocessor = ModellingPreprocessor(target=response_variable)
        preprocessor.fit(data)
    elif preprocessor is not None and not preprocessor.is_fitted:
        preprocessor.fit(data)

    df_train_pre = preprocessor.transform(df_train)
    df_test_pre = preprocessor.transform(df_test)

    return df_train_pre, df_test_pre


def sample_split_data(data: pd.DataFrame, response_variable: str, explanatory_variables: Optional[List[str]] = None,
                      test_size: float = 0.2, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if explanatory_variables is not None:
        data = data[np.concatenate((explanatory_variables, [response_variable]))]

    if sample_size is not None and sample_size < len(data):
        data = data.sample(sample_size, random_state=RANDOM_SEED)

    if all(data[response_variable].value_counts().values > 1) and data[response_variable].nunique() <= sqrt(len(data)):
        df_train, df_test = train_test_split(data, test_size=test_size, stratify=data[response_variable],
                                             random_state=RANDOM_SEED)
    else:
        df_train, df_test = train_test_split(data, test_size=test_size, random_state=RANDOM_SEED)

        train_unique = df_train[response_variable].unique()
        target_in_train = df_test[response_variable].apply(lambda y: True if y in train_unique else False)
        df_test = df_test[target_in_train]

    return df_train, df_test


def _check_model_type(model: Union[str, ClassifierMixin, RegressorMixin], copy_model: bool,
                      task: str) -> Union[ClassifierMixin, RegressorMixin]:
    if task == 'clf':
        if isinstance(model, str):
            if model in CLASSIFIERS:
                estimator = CLASSIFIERS[model]()
            else:
                raise KeyError(f"Selected model '{model}' not available for CLASSIFICATION.")
        elif isinstance(model, BaseEstimator):
            if not isinstance(model, ClassifierMixin):
                raise KeyError("Given model is not available for CLASSIFICATION.")

            if copy_model:
                estimator = clone(model)
            else:
                estimator = model
        else:
            raise ValueError("Given model type is not compatible")

    elif task == 'rgr':
        if isinstance(model, str):
            if model in REGRESSORS:
                estimator = REGRESSORS[model]()
            else:
                raise KeyError(f"Selected model '{model}' not available for REGRESSION.")
        elif isinstance(model, BaseEstimator):
            if not isinstance(model, RegressorMixin):
                raise KeyError("Given model is not available for REGRESSION.")

            if copy_model:
                estimator = clone(model)
            else:
                estimator = model
        else:
            raise ValueError("Given model type is not compatible")

    else:
        raise ValueError(f"Given task '{task}' not recognized")

    return estimator
