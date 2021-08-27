import logging
from math import sqrt
from typing import Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

logger = logging.getLogger(__name__)

"""A dictionary of sklearn classifiers with fit/predict methods."""
CLASSIFIERS: Dict[str, Type[ClassifierMixin]] = {
    'Linear': RidgeClassifier,
    'Logistic': LogisticRegression,
    'GradientBoosting': GradientBoostingClassifier,
    'RandomForest': RandomForestClassifier,
    'MLP': MLPClassifier,
    'LinearSVM': LinearSVC
}

"""A dictionary of sklearn regressors with fit/predict methods."""
REGRESSORS: Dict[str, Type[RegressorMixin]] = {
    'Linear': Ridge,
    'GradientBoosting': GradientBoostingRegressor,
    'RandomForest': RandomForestRegressor,
    'MLP': MLPRegressor,
    'LinearSVM': LinearSVR
}

RANDOM_SEED = 42


def sample_split_data(df: pd.DataFrame,
                      response_variable: str,
                      explanatory_variables: Optional[List[str]] = None,
                      test_size: Optional[float] = 0.2,
                      sample_size: Optional[int] = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if explanatory_variables is not None:
        df = cast(pd.DataFrame, df[np.concatenate((explanatory_variables, [response_variable]))])

    if sample_size is not None and sample_size < len(df):
        df = cast(pd.DataFrame, df.sample(sample_size, random_state=RANDOM_SEED))

    if df[response_variable].nunique() > sqrt(len(df)):
        df_train, df_test = train_test_split(df,
                                             test_size=test_size,
                                             random_state=RANDOM_SEED)
    elif all(df[response_variable].value_counts().values > 1)\
            and df[response_variable].nunique() <= sqrt(len(df)):
        df_train, df_test = train_test_split(df,
                                             test_size=test_size,
                                             stratify=df[response_variable],
                                             random_state=RANDOM_SEED)
    else:
        df_train, df_test = train_test_split(df,
                                             test_size=test_size,
                                             random_state=RANDOM_SEED)

        train_unique = df_train[response_variable].unique()
        target_in_train = df_test[response_variable]\
            .apply(lambda y: True if y in train_unique else False)
        df_test = df_test[target_in_train]

    return df_train, df_test


def _get_regression_model(model: Union[str, RegressorMixin],
                          copy_model: bool) -> RegressorMixin:
    estimator = None
    if isinstance(model, str):
        if model in REGRESSORS:
            estimator = REGRESSORS[model]()
        else:
            raise KeyError(f"Selected model '{model}'\
                                 not available for REGRESSION.")
    elif isinstance(model, BaseEstimator):
        if not isinstance(model, RegressorMixin):
            raise KeyError("Given model is not available for REGRESSION.")

        if copy_model:
            estimator = clone(model)
        else:
            estimator = model
    else:
        raise ValueError("Given model type is not compatible")

    return estimator


def _get_classification_model(model: Union[str, ClassifierMixin],
                              copy_model: bool) -> ClassifierMixin:
    estimator = None
    if isinstance(model, str):
        if model in CLASSIFIERS:
            estimator = CLASSIFIERS[model]()
        else:
            raise KeyError(f"Selected model '{model}'\
                                 not available for CLASSIFICATION.")
    elif isinstance(model, BaseEstimator):
        if not isinstance(model, ClassifierMixin):
            raise KeyError("Given model is not available for CLASSIFICATION.")

        if copy_model:
            estimator = clone(model)
        else:
            estimator = model
    else:
        raise ValueError("Given model type is not compatible")
    return estimator


def check_model_type(model: Union[str, ClassifierMixin, RegressorMixin],
                     copy_model: bool,
                     task: str) -> Union[ClassifierMixin, RegressorMixin]:
    """Given a model (string or type) and task (classification or regreesion),
     return the estimator corresponding to that model"""
    if task == 'clf':
        estimator = _get_classification_model(model=model, copy_model=copy_model)
    elif task == 'rgr':
        estimator = _get_regression_model(model=model, copy_model=copy_model)
    else:
        raise ValueError(f"Given task '{task}' not recognized")

    return estimator
