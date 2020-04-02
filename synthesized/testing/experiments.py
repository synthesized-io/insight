from typing import Optional, Dict, List, Tuple, Any

import simplejson
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    precision_recall_curve, confusion_matrix
from sklearn.base import clone
from scipy.signal import filtfilt

# Set the style of plots
plt.style.use('seaborn')
mpl.rcParams["axes.facecolor"] = 'w'
mpl.rcParams['grid.color'] = 'grey'
mpl.rcParams['grid.alpha'] = 0.1

mpl.rcParams['axes.linewidth'] = 0.3
mpl.rcParams['axes.edgecolor'] = 'grey'

mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.top'] = True


class ExperimentalEstimator:
    def __init__(self, target: str):
        self.target = target
        self.oh_encoders: Dict[str, Optional[Tuple[OneHotEncoder, List[str]]]] = dict()

    def preprocess_df(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        xx = []
        c_names: List[str] = []

        # Train OH Encoders
        if len(self.oh_encoders) == 0:
            for c in data.columns:
                column = data[c]
                if column.dtype.kind not in ('f', 'i') or column.nunique() <= 2:
                    # print("Converting '{}' to One-Hot encoding".format(c))
                    if column.nunique() > 1:
                        oh = OneHotEncoder(drop='first')
                        x_i = oh.fit_transform(column.values.reshape(-1, 1)).todense()
                        names = ['{}_{}'.format(c, enc) for enc in oh.categories_[0][1:]]
                    else:
                        oh = OneHotEncoder()
                        x_i = oh.fit_transform(column.values.reshape(-1, 1)).todense()
                        names = [c]
                    if len(names) > 1:
                        c_names = list(np.concatenate((c_names, names)))
                        self.oh_encoders[c] = (oh, names)
                    else:
                        c_names = list(np.concatenate((c_names, [c])))
                        self.oh_encoders[c] = (oh, [c])
                else:
                    x_i = column.values.reshape(-1, 1)
                    c_names = list(np.concatenate((c_names, [c])))
                    self.oh_encoders[c] = None
                xx.append(x_i)

        # OH Encoders are already trained
        elif len(self.oh_encoders) > 0:
            for c in data.columns:
                column = data[c]
                oh_encoder_names = self.oh_encoders[c]
                if oh_encoder_names is not None:
                    oh_encoder, names = oh_encoder_names
                    x_i = oh_encoder.transform(column.values.reshape(-1, 1)).todense()
                    c_names = list(np.concatenate((c_names, names)))
                else:
                    x_i = column.values.reshape(-1, 1)
                    c_names = list(np.concatenate((c_names, [c])))
                xx.append(x_i)

        xx = np.hstack(xx)
        return pd.DataFrame(xx, columns=c_names)

    def classification(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                       classifiers: Dict[str, BaseEstimator] = None, name: str = None,
                       results: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:

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

        features = list(filter(lambda x: x != self.target, df_train.columns))

        X_train = df_train[features].astype(np.float64).values
        y_train = df_train[self.target].astype(np.float64).values
        X_test = df_test[features].astype(np.float64).values
        y_test = df_test[self.target].astype(np.float64).values

        for clf_name, clf_ in classifiers.items():
            clf = clone(clf_)

            clf.fit(X_train, y_train)
            if hasattr(clf, 'predict_proba'):
                f_prob_train = clf.predict_proba(X_train).T[1]
                f_train = [0 if p < 0.5 else 1 for p in f_prob_train]
                f_prob_test = clf.predict_proba(X_test).T[1]
                f_test = [0 if p < 0.5 else 1 for p in f_prob_test]
            else:
                f_prob_train = f_train = clf.predict(X_train)
                f_prob_test = f_test = clf.predict(X_test)

            results.append({
                'Classifier': clf_name,
                'Name': name,

                # Train
                'Accuracy Train': accuracy_score(y_train, f_train),
                'Precision Train': precision_score(y_train, f_train),
                'Recall Train': recall_score(y_train, f_train),
                'F1 Train': f1_score(y_train, f_train),
                'AUC Train': roc_auc_score(y_train, f_prob_train),

                # Test
                'Accuracy Test': accuracy_score(y_test, f_test),
                'Precision Test': precision_score(y_test, f_test),
                'Recall Test': recall_score(y_test, f_test),
                'F1 Test': f1_score(y_test, f_test),
                'AUC Test': roc_auc_score(y_test, f_prob_test),

                'ROC Curve': roc_curve(y_test, f_prob_test),
                'PR Curve': precision_recall_curve(y_test, f_prob_test),
                'Confusion Matrix': confusion_matrix(y_test, f_test)
            })

        return results

    def preprocess_classify(self, original_data, other_dfs: Dict[str, pd.DataFrame], train_idx: np.array,
                            test_idx: np.array, name_orig: Optional[str] = 'Original',
                            classifiers: Dict[str, BaseEstimator] = None) -> pd.DataFrame:

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
            print("\nClassifying original data-frame...")
            results = self.classification(train, test, name=name_orig, results=results, classifiers=classifiers)

        for name, df in other_dfs.items():
            print("\nClassifying data-frame '{}'...".format(name))
            results = self.classification(df, test, name=name, results=results, classifiers=classifiers)

        df_results = pd.DataFrame(results)
        return df_results


def read_data_run_experiment(func, synthesized_path, config_file, func_kwargs=None):
    if func_kwargs is None:
        func_kwargs = {}

    j = simplejson.load(open(synthesized_path + config_file, 'r'))
    results = dict()

    for name, config in j['instances'].items():
        print("\n>> Computing dataset '{}' <<".format(name))
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
