import logging
from math import sqrt
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

from .preprocessor import ModellingPreprocessor

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

RANDOM_SEED = 42


def sample_split_data(df: pd.DataFrame,
                      response_variable: str,
                      explanatory_variables: Optional[List[str]] = None,
                      test_size: float = 0.2,
                      sample_size: Optional[int] = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if explanatory_variables is not None:
        df = df[np.concatenate((explanatory_variables, [response_variable]))]

    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=RANDOM_SEED)

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


def preprocess_split_data(df: pd.DataFrame,
                          response_variable: str,
                          explanatory_variables: Optional[List[str]] = None,
                          test_size: float = 0.2,
                          sample_size: Optional[int] = None,
                          preprocessor: ModellingPreprocessor = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = sample_split_data(df,
                                          response_variable,
                                          explanatory_variables,
                                          test_size=test_size,
                                          sample_size=sample_size)

    if preprocessor is None:
        preprocessor = ModellingPreprocessor(target=response_variable)
        preprocessor.fit(df)
    elif preprocessor is not None and not preprocessor.is_fitted:
        preprocessor.fit(df)

    df_train_pre = preprocessor.transform(df_train)
    df_test_pre = preprocessor.transform(df_test)

    return df_train_pre, df_test_pre


def check_model_type(model: Union[str, ClassifierMixin, RegressorMixin],
                     copy_model: bool,
                     task: str) -> Union[ClassifierMixin, RegressorMixin]:
    if task == 'clf':
        if isinstance(model, str):
            if model in CLASSIFIERS:
                estimator = CLASSIFIERS[model]()
            else:
                raise KeyError(f"Selected model '{model}'\
                                 not available for CLASSIFICATION.")
        elif isinstance(model, BaseEstimator):
            if not isinstance(model, ClassifierMixin):
                raise KeyError("Given model is not\
                                available for CLASSIFICATION.")

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

    else:
        raise ValueError(f"Given task '{task}' not recognized")

    return estimator