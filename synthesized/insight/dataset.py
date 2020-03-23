from itertools import chain
from typing import Dict, Callable, Type, Union, List

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    value_counts = describe_dataset_values(df).groupby('class_name').size().to_dict()

    properties = {f'num_{value}': count for value, count in value_counts.items()}
    properties['total_rows'] = len(df)
    properties['total_columns'] = sum([n for v, n in value_counts.items() if v != 'NanValue'])

    return pd.DataFrame.from_records([properties]).T.reset_index().rename(columns={'index': 'property', 0: 'value'})


def classification_score(df_train: pd.DataFrame, df_test: pd.DataFrame,
                         label: str, model: str = None, models: List[str] = None) -> pd.DataFrame:

    if model is not None:
        if model == 'all':
            models = [*CLASSIFICATION_MODELS.keys()]
        else:
            models = [model]

    if models is None:
        raise ValueError

    for model in models:
        if model not in CLASSIFICATION_MODELS:
            raise ValueError

    vf = ValueFactory(df=pd.concat((df_train, df_test), axis='index'))
    train, test = vf.preprocess(df_train), vf.preprocess(df_test)
    train_label = train.pop(label).to_numpy()
    train_features = train.to_numpy()
    test_label = test.pop(label).to_numpy()
    test_features = test.to_numpy()

    records = []
    for model in models:
        clf = CLASSIFICATION_MODELS[model]()
        clf.fit(X=train_features, y=train_label)
        predicted_label = clf.predict(X=test_features)

        metrics: Dict[str, Union[str, float]] = dict(model=model)
        for name, metric in METRICS.items():
            metrics[name] = metric(test_label, predicted_label)

        records.append(metrics)
    df_metrics = pd.DataFrame.from_records(records)

    return df_metrics
