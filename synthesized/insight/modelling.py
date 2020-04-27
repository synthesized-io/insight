from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

from ..common import ValueFactory


CLASSIFIERS: Dict[str, Type[ClassifierMixin]] = {
    'Linear': RidgeClassifier,
    'Logistic': LogisticRegression,
    'GradientBoosting': GradientBoostingClassifier,
    'RandomForest': RandomForestClassifier,
    'MLP': MLPClassifier,
    'LinearSVC': LinearSVC
}
"""A dictionary of sklearn classifiers with fit/predict methods."""

REGRESSORS: Dict[str, Type[RegressorMixin]] = {
    'Linear': Ridge,
    'GradientBoosting': GradientBoostingRegressor,
    'RandomForest': RandomForestRegressor,
    'MLP': MLPRegressor,
    'LinearSVC': LinearSVR
}
"""A dictionary of sklearn regressors with fit/predict methods."""


def r2_regression_score(df_train: pd.DataFrame, df_test: pd.DataFrame, regressor: str,
                        y_column: str, x_columns: List[str] = None) -> float:
    vf = ValueFactory(df=pd.concat((df_train, df_test), axis='index'))
    train, test = vf.preprocess(df_train).dropna(), vf.preprocess(df_test).dropna()

    X_train, y_train = train[x_columns], train[y_column].values
    X_test, y_test = test[x_columns], test[y_column].values

    rgr = REGRESSORS[regressor]
    rgr.fit(X_train, y_train)

    f_test = rgr.predict(X_test)

    y_val = [val for val in vf.values if val.name == y_column][0]

    y_test = y_val.postprocess(pd.DataFrame({y_column: y_test})).values
    f_test = y_val.postprocess(pd.DataFrame({y_column: f_test})).values

    return r2_score(y_test, f_test)


def roc_auc_classification_score(train: pd.DataFrame, test: pd.DataFrame, classifier: str,
                                 y_column: str, x_columns: List[str] = None) -> float:

    # If a target group is in test but not in train, it will have problems with shapes (OH.shape != CLF.shape).
    train_unique = train[y_column].unique()
    target_in_train = test[y_column].apply(lambda y: True if y in train_unique else False)
    test = test[target_in_train]

    train = train.dropna()
    test = test.dropna()

    x_train, y_train = train[x_columns], train[y_column].values
    x_test, y_test = test[x_columns], test[y_column].values

    if len(np.unique(y_train)) == 1:
        return 1.

    x_train, x_test = _preprocess_x(x_train, x_test)
    clf = CLASSIFIERS[classifier]()
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


def logistic_regression_r2(df, y_label: str, x_labels: List[str]):
    rg = CLASSIFIERS['Logistic']()
    rg.fit(df[x_labels].to_numpy().reshape((-1, 1)), df[y_label])

    labels = df[y_label].map({c: n for n, c in enumerate(rg.classes_)}).to_numpy()
    oh_labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))

    lp = rg.predict_log_proba(df['y'].to_numpy().reshape((-1, 1)))
    llf = np.sum(oh_labels * lp)

    rg = LogisticRegression()
    rg.fit(np.ones(df[x_labels].to_numpy().reshape((-1, 1)).shape), df[y_label])

    lp = rg.predict_log_proba(df[x_labels].to_numpy().reshape((-1, 1)))
    llnull = np.sum(oh_labels * lp)

    psuedo_r2 = 1 - (llf / llnull)

    return psuedo_r2


def _preprocess_x(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[np.array, np.array]:
    columns_categorical = []
    columns_numeric = []
    vf = ValueFactory(df=pd.concat((x_train, x_test), axis='index'))

    plot_type_by_columns = {val.name: type(val).__name__ for val in vf.values}
    for column in x_train.columns.values:
        if plot_type_by_columns[column] == 'CategoricalValue':
            columns_categorical.append(column)
        else:
            columns_numeric.append(column)

    pt = StandardScaler()
    oh = OneHotEncoder(categories='auto', sparse=False)

    train_result = []
    test_result = []
    if len(columns_numeric) > 0:
        pt.fit(np.concatenate([x_train[columns_numeric], x_test[columns_numeric]]))
        train_result.append(pt.transform(x_train[columns_numeric]))
        test_result.append(pt.transform(x_test[columns_numeric]))

    if len(columns_categorical) > 0:
        oh.fit(np.concatenate([x_train[columns_categorical], x_test[columns_categorical]]))
        train_result.append(oh.transform(x_train[columns_categorical]))
        test_result.append(oh.transform(x_test[columns_categorical]))

    return np.concatenate(train_result, axis=1), np.concatenate(test_result, axis=1)


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
