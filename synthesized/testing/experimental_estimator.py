import logging
from typing import Optional, Dict, List, Any, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simplejson
from scipy.signal import filtfilt
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    precision_recall_curve, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from ..metadata import MetaExtractor, DataFrameMeta, ContinuousMeta,  CategoricalMeta, DateMeta

# Set the style of plots
plt.style.use('seaborn')
mpl.rcParams["axes.facecolor"] = 'w'
mpl.rcParams['grid.color'] = 'grey'
mpl.rcParams['grid.alpha'] = 0.1

mpl.rcParams['axes.linewidth'] = 0.3
mpl.rcParams['axes.edgecolor'] = 'grey'

mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.top'] = True

logger = logging.getLogger(__name__)


class ExperimentalEstimator:
    def __init__(self, target: str):
        self.target = target
        # self.oh_encoders: Dict[str, Optional[Tuple[OneHotEncoder, List[str]]]] = dict()
        self.oh_encoders: Dict[str, Union[LabelEncoder, OneHotEncoder]] = dict()
        self.all_metrics = [
            'Accuracy',
            'Precision',
            'Recall',
            'F1',
            'AUC',
            'ROC_Curve',
            'PR_Curve',
            'ConfusionMatrix'
        ]
        self.dp: Optional[DataFrameMeta] = None

    def preprocess_df(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        self.dp = MetaExtractor.extract(data)

        xx = []
        c_names: List[str] = []

        for v in self.dp.values:
            column = data[v.name]
            x_i = c_name_i = None

            if v.name == self.target:
                x_i = column.values.reshape(-1, 1)
                if isinstance(v, CategoricalMeta):
                    if v.name not in self.oh_encoders:
                        self.oh_encoders[v.name] = LabelEncoder()
                        self.oh_encoders[v.name].fit(x_i)
                    x_i = self.oh_encoders[v.name].transform(x_i).reshape(-1, 1)
                else:
                    x_i = pd.to_numeric(column, errors='coerce').values.reshape(-1, 1)

                c_name_i = [v.name]
            elif isinstance(v, DateMeta):
                x_i = pd.to_numeric(pd.to_datetime(column), errors='coerce').values.reshape(-1, 1)
                c_name_i = [v.name]
            elif isinstance(v, ContinuousMeta):
                x_i = column.values.reshape(-1, 1)
                c_name_i = [v.name]
            elif isinstance(v, CategoricalMeta):
                x_i = column.values.reshape(-1, 1)
                if v.name not in self.oh_encoders:
                    self.oh_encoders[v.name] = OneHotEncoder(drop='first', sparse=False)
                    self.oh_encoders[v.name].fit(x_i)

                x_i = self.oh_encoders[v.name].transform(column.values.reshape(-1, 1))
                c_name_i = ['{}_{}'.format(v.name, enc) for enc in self.oh_encoders[v.name].categories_[0][1:]]

            else:
                logger.debug(f"Ignoring column {v.name} (type {v.__class__.__name__})")

            if x_i is not None and c_name_i is not None:
                xx.append(x_i)
                c_names.extend(c_name_i)

        xx = np.hstack(xx)
        return pd.DataFrame(xx, columns=c_names)

    def classification(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                       classifiers: Dict[str, BaseEstimator] = None, name: str = None,
                       results: List[Dict[str, Any]] = None, copy_clf: bool = True,
                       metrics_to_compute: List[str] = None) -> List[Dict[str, Any]]:

        if classifiers is None:
            classifiers = {
                'LogisticRegression': RidgeClassifier(max_iter=500),
                'SVM': LinearSVC(max_iter=1000),
                'RandomForest': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=1000)
            }

        if results is None:
            results = []

        if metrics_to_compute is None:
            metrics_to_compute = self.all_metrics

        features = list(filter(lambda x: x != self.target, df_train.columns))

        X_train = df_train[features].astype(np.float64).values
        y_train = df_train[self.target].astype(np.float64).values
        X_test = df_test[features].astype(np.float64).values
        y_test = df_test[self.target].astype(np.float64).values

        if len(np.unique(y_train)) == 1 or len(np.unique(y_test)) == 1:
            logger.info("A single class found in target column. Returning empty List.")
            return results

        for clf_name, clf_ in classifiers.items():
            clf = clone(clf_) if copy_clf else clf_

            # Check if the classifiers have been trained.
            try:
                check_is_fitted(clf)
            except NotFittedError:
                clf.fit(X_train, y_train)

            y_test_oh = y_test
            if hasattr(clf, 'predict_proba'):
                f_prob_test = clf.predict_proba(X_test)
                f_test = np.argmax(f_prob_test, axis=1)

                # ROC AUC would fail if one class is not present
                if f_prob_test.shape[1] == 2:
                    f_prob_test = f_prob_test.T[1]
                elif f_prob_test.shape[1] > len(np.unique(y_test)):
                    f_prob_test = f_prob_test[:, np.unique(y_test).astype(int)]
                    oh = OneHotEncoder(sparse=False)
                    y_test_oh = oh.fit_transform(y_test.reshape(-1, 1))

            else:
                logger.warning(f"Given classifier '{clf_name}' doesn't have 'predict_proba' attr, needed to compute "
                               f"some metrics.")
                f_prob_test = f_test = clf.predict(X_test)

            results_i = {
                'Classifier': clf_name,
                'Name': name,
                'PredictedValues': f_prob_test
            }

            if 'Accuracy' in metrics_to_compute:
                results_i['Accuracy'] = accuracy_score(y_test, f_test)
            if 'Precision' in metrics_to_compute:
                results_i['Precision'] = precision_score(y_test, f_test, average='micro')
            if 'Recall' in metrics_to_compute:
                results_i['Recall'] = recall_score(y_test, f_test, average='micro')
            if 'F1' in metrics_to_compute:
                results_i['F1'] = f1_score(y_test, f_test, average='micro')
            if 'AUC' in metrics_to_compute:
                results_i['AUC'] = roc_auc_score(y_test_oh, f_prob_test, multi_class='ovo')

            if 'ROC_Curve' in metrics_to_compute:
                results_i['ROC_Curve'] = roc_curve(y_test_oh, f_prob_test) \
                    if len(f_prob_test) == 1 else None
            if 'PR_Curve' in metrics_to_compute:
                results_i['PR_Curve'] = precision_recall_curve(y_test_oh, f_prob_test) \
                    if len(f_prob_test) == 1 else None
            if 'ConfusionMatrix' in metrics_to_compute:
                results_i['ConfusionMatrix'] = confusion_matrix(y_test, f_test)

            results.append(results_i)

        return results

    def preprocess_classify(self, original_data: pd.DataFrame, other_dfs: Dict[str, pd.DataFrame],
                            train_idx: np.ndarray, test_idx: np.ndarray, name_orig: Optional[str] = 'Original',
                            classifiers: Dict[str, BaseEstimator] = None, metrics_to_compute: List[str] = None,
                            copy_clf: bool = True) -> pd.DataFrame:

        original_data = original_data.copy()
        other_dfs = other_dfs.copy()

        # Pre-process
        original_data = self.preprocess_df(original_data)
        for name, df in other_dfs.items():
            other_dfs[name] = self.preprocess_df(df)

        train = original_data.iloc[train_idx]
        test = original_data.iloc[test_idx]

        # Make classification
        results: List[Dict[str, Any]] = []

        if name_orig:
            logger.info("Classifying original data-frame...")
            results = self.classification(train, test, name=name_orig, results=results, classifiers=classifiers,
                                          metrics_to_compute=metrics_to_compute, copy_clf=copy_clf)

        for name, df in other_dfs.items():
            logger.info("Classifying data-frame '{}'...".format(name))
            results = self.classification(df, test, name=name, results=results, classifiers=classifiers,
                                          metrics_to_compute=metrics_to_compute, copy_clf=copy_clf)

        df_results = pd.DataFrame(results)
        return df_results


def read_data_run_experiment(func, synthesized_path, config_file, func_kwargs=None):
    if func_kwargs is None:
        func_kwargs = {}

    j = simplejson.load(open(synthesized_path + config_file, 'r'))
    results = dict()

    for name, config in j['instances'].items():
        logger.info("\n>> Computing dataset '{}' <<".format(name))
        data = pd.read_csv(synthesized_path + config['data'])
        if 'ignore_columns' in config.keys():
            data.drop(config['ignore_columns'], axis=1, inplace=True)

        kwargs = {k: v for k, v in config.items() if k not in ('data', 'ignore_columns', 'num_passes')}

        func_kwargs = dict(func_kwargs, **kwargs)
        func_kwargs['file_name'] = '{}.png'.format(name)

        results[name] = func(data=data, **func_kwargs)
    return results


def get_lowfreq_trend(y):
    y = np.array(y)

    b_n = int(np.ceil(len(y) / 6))
    b = [1. / b_n] * b_n
    a = 1
    y_low = filtfilt(b, a, y)

    return y_low


def get_linear_trend(y, x=None):
    if x is None:
        x = np.array(range(len(y)))
    xx = np.vstack([x, np.ones(len(y))]).T
    m, n = np.linalg.lstsq(xx, y, rcond=None)[0]

    return m * x + n


def plot_results_curve(d, datset, classifier, test_metric, ax):
    columns = d[datset].columns

    keep_cols = list(filter(lambda x: x.startswith(classifier), columns))

    df_results = d[datset][keep_cols].copy()

    x = []
    for name in df_results.loc[test_metric].index:
        _, frac = name.split('_', 1)
        x.append(int(frac))

    X_sorted = np.array([[x, y] for y, x in sorted(zip(x, df_results.loc[test_metric].values))]).T
    x, y = X_sorted[1], X_sorted[0]

    ax.plot(x, y, label=test_metric)
    ax.plot([100, 100], [min(y), max(y)], 'k:')
    ax.plot(x, get_linear_trend(y), 'k--')
    ax.set_ylabel(test_metric)
    ax.set_xlabel('Sampling Proportion')
