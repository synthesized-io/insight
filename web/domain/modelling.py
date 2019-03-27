import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics.scorer import r2_scorer, roc_auc_scorer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from .dataset_meta import DatasetMeta, DENSITY_PLOT_TYPE


def r2_regression_score(regressor: RegressorMixin, train: pd.DataFrame, test: pd.DataFrame, meta: DatasetMeta, y_column: str):
    train = train.dropna()
    test = test.dropna()

    X_columns = list(train.columns.values)
    X_columns.remove(y_column)

    X_train, y_train = train[X_columns], train[y_column]
    X_test, y_test = test[X_columns], test[y_column]

    X_train, X_test = _preprocess_X(X_train, X_test, meta)

    regressor.fit(X_train, y_train)

    return r2_scorer(regressor, X_test, y_test)


def roc_auc_classification_score(classifier: ClassifierMixin, train: pd.DataFrame, test: pd.DataFrame, meta: DatasetMeta, y_column: str):
    train = train.dropna()
    test = test.dropna()

    X_columns = list(train.columns.values)
    X_columns.remove(y_column)

    X_train, y_train = train[X_columns], train[y_column]
    X_test, y_test = test[X_columns], test[y_column]

    X_train, X_test = _preprocess_X(X_train, X_test, meta)

    classifier.fit(X_train, y_train)

    return roc_auc_scorer(classifier, X_test, y_test)


def _preprocess_X(X_train: pd.DataFrame, X_test: pd.DataFrame, meta: DatasetMeta):
    columns_to_encode = []
    columns_to_scale = []
    plot_type_by_columns = {col.name: col.plot_type for col in meta.columns}
    for column in X_train.columns.values:
        if plot_type_by_columns[column] == DENSITY_PLOT_TYPE:
            columns_to_scale.append(column)
        else:
            columns_to_encode.append(column)

    pt = PowerTransformer()
    oh = OneHotEncoder(sparse=False)

    train_result = []
    test_result = []
    if len(columns_to_scale) > 0:
        pt.fit(np.concatenate([X_train[columns_to_scale], X_test[columns_to_scale]]))
        train_result.append(pt.transform(X_train[columns_to_scale]))
        test_result.append(pt.transform(X_test[columns_to_scale]))
    if len(columns_to_encode) > 0:
        oh.fit(np.concatenate([X_train[columns_to_encode], X_test[columns_to_encode]]))
        train_result.append(oh.transform(X_train[columns_to_encode]))
        test_result.append(oh.transform(X_test[columns_to_encode]))

    return np.concatenate(train_result, axis=1), np.concatenate(test_result, axis=1)
