from itertools import chain
from typing import Dict, Callable, Type, Union

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from ..common import ValueFactory


METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'roc_auc': roc_auc_score
}

CLASSIFICATION_MODELS: Dict[
    str,
    Type[Union[LogisticRegression, GradientBoostingClassifier, RandomForestClassifier, MLPClassifier, SVC, LinearSVC]]
] = {
    'LogisticRegression': LogisticRegression,
    'GradientBoosting': GradientBoostingClassifier,
    'RandomForest': RandomForestClassifier,
    'MLP': MLPClassifier,
    'SVC': SVC,
    'LinearSVC': LinearSVC
}
"""A dictionary of sklearn classifiers with fit/predict methods."""


def describe_dataset_values(df: pd.DataFrame) -> pd.DataFrame:
    vf = ValueFactory(df=df)
    values = vf.get_values()

    value_spec = [
        {k: j for k, j in chain(v.specification().items(), [('class_name', v.__class__.__name__)])}
        for v in values
    ]
    for s in value_spec:
        if 'categories' in s:
            s['categories'] = '[' + ', '.join([str(o) for o in s['categories']]) + ']'

    for n, v in enumerate(values):
        if hasattr(v, 'day'):
            value_spec[n]['embedding_size'] = v.learned_input_size()
        if v.__class__.__name__ == 'NanValue':
            value_spec.append(
                {'class_name': v.value.__class__.__name__, 'name': v.name + '_value'})

    df_values = pd.DataFrame.from_records(value_spec)

    return df_values


def classification_score(df: pd.DataFrame, label: str, model: str) -> pd.DataFrame:
    if model not in CLASSIFICATION_MODELS:
        raise ValueError

    vf = ValueFactory(df=df)
    df = vf.preprocess(df)
    clf = CLASSIFICATION_MODELS[model]

    test, train = train_test_split(df, test_size=0.2)

    train_label = train.pop(label).to_numpy()
    train_features = train.to_numpy()
    test_label = test.pop(label).to_numpy()
    test_features = test.to_numpy()

    clf.fit(X=train_features, y=train_label)
    predicted_label = clf.predict(X=test_features)

    df_metrics = pd.DataFrame({
        name: metric(test_label, predicted_label)
        for name, metric in METRICS.items()
    })

    return df_metrics
