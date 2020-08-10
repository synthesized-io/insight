import logging
from typing import Any, Union, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    precision_recall_curve, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
import seaborn as sns

from .preprocessor import ModellingPreprocessor
from ...metadata import MetaExtractor

logger = logging.getLogger(__name__)

REGRESSION_METRICS = ['mean_absolute_error', 'mean_squared_error', 'r2_score']
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
CLASSIFICATION_PLOT_METRICS = ['roc_curve', 'pr_curve', 'confusion_matrix']

DEFAULT_REGRESSION_METRICS = ['mean_squared_error', 'r2_score']
DEFAULT_CLASSIFICATION_METRICS = ['accuracy', 'f1_score', 'roc_auc']
DEFAULT_CLASSIFICATION_PLOT_METRICS = ['roc_curve', 'pr_curve']


def classifier_scores_from_df(df_train: Optional[pd.DataFrame], df_test: pd.DataFrame,
                              target: str, clf: ClassifierMixin,
                              metrics: Optional[Union[str, List[str]]] = 'roc_auc') -> Dict[str, Any]:
    x_labels = list(filter(lambda v: v != target, df_test.columns))

    if df_train is not None:
        x_train = df_train[x_labels].to_numpy()
        y_train = df_train[target].to_numpy()
    else:
        x_train = y_train = None
    x_test = df_test[x_labels].to_numpy()
    y_test = df_test[target].to_numpy()

    return classifier_scores(x_train, y_train, x_test, y_test, clf=clf, metrics=metrics)


def regressor_scores_from_df(df_train: Optional[pd.DataFrame], df_test: pd.DataFrame,
                             target: str, rgr: RegressorMixin,
                             metrics: Optional[Union[str, List[str]]] = 'r2_score') -> Dict[str, Any]:
    x_labels = list(filter(lambda v: v != target, df_test.columns))

    if df_train is not None:
        x_train = df_train[x_labels].to_numpy()
        y_train = df_train[target].to_numpy()
    else:
        x_train = y_train = None
    x_test = df_test[x_labels].to_numpy()
    y_test = df_test[target].to_numpy()

    return regressor_scores(x_train, y_train, x_test, y_test, rgr=rgr, metrics=metrics)


def plot_metrics_from_df(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str, clf: ClassifierMixin,
                         metrics: Optional[Union[str, List[str]]] = None,
                         axes: Optional[Union[plt.Axes, List[plt.Axes]]] = None, name: str = None):
    x_labels = list(filter(lambda v: v != target, df_train.columns))
    x_train = df_train[x_labels].to_numpy()
    y_train = df_train[target].to_numpy()
    x_test = df_test[x_labels].to_numpy()
    y_test = df_test[target].to_numpy()

    plot_metrics(x_train, y_train, x_test, y_test, clf=clf, metrics=metrics, axes=axes, name=name)


def classifier_scores(x_train: Optional[np.ndarray], y_train: Optional[np.ndarray],
                      x_test: np.ndarray, y_test: np.ndarray,
                      clf: ClassifierMixin, metrics: Optional[Union[str, List[str]]] = 'roc_auc') -> Dict[str, Any]:

    if isinstance(metrics, str):
        metrics = [metrics]
    elif metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS

    all_classification_metrics = np.concatenate((CLASSIFICATION_METRICS, CLASSIFICATION_PLOT_METRICS))
    missing_metrics = list(filter(lambda m: m not in all_classification_metrics, metrics))
    if len(missing_metrics) > 0:
        raise ValueError("Can't compute following metrics: '{}'".format("', '".join(missing_metrics)))

    # Single class present in target
    if len(np.unique(y_train)) == 1:
        return {metric: 1. for metric in metrics}

    # Check if fitted
    try:
        check_is_fitted(clf)
    except NotFittedError:
        if x_train is None or y_train is None:
            raise ValueError("'x_train' and 'y_train' must be given if the classifier has not been trained")
        clf.fit(x_train, y_train)

    results: Dict[str, Any] = dict()

    # Two classes classification
    if len(np.unique(y_train)) == 2:
        if hasattr(clf, 'predict_proba'):
            f_proba_test = clf.predict_proba(x_test)
            y_pred_test = np.argmax(f_proba_test, axis=1)
            f_test = f_proba_test.T[1]
        else:
            y_pred_test = f_test = clf.predict(x_test)

        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_test, y_pred_test)
        if 'precision' in metrics:
            results['precision'] = precision_score(y_test, y_pred_test)
        if 'recall' in metrics:
            results['recall'] = recall_score(y_test, y_pred_test)
        if 'f1_score' in metrics:
            results['f1_score'] = f1_score(y_test, y_pred_test)
        if 'roc_auc' in metrics:
            results['roc_auc'] = roc_auc_score(y_test, f_test)

        # Plot metrics
        if 'roc_curve' in metrics:
            results['roc_curve'] = roc_curve(y_test, f_test)
        if 'pr_curve' in metrics:
            results['pr_curve'] = precision_recall_curve(y_test, f_test)
        if 'confusion_matrix' in metrics:
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)

    # Multi-class classification
    else:
        oh = OneHotEncoder(sparse=False)
        oh.fit(np.concatenate((y_train, y_test)).reshape(-1, 1))
        y_test_oh = oh.transform(y_test.reshape(-1, 1))

        if hasattr(clf, 'predict_proba'):
            f_proba_test = clf.predict_proba(x_test)
            y_pred_test = np.argmax(f_proba_test, axis=1)
        else:
            y_pred_test = clf.predict(x_test)
            f_proba_test = oh.transform(y_pred_test.reshape(-1, 1)).toarray()

        y_test_oh, f_proba_test = _remove_zero_column(y_test_oh, f_proba_test)

        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_test, y_pred_test)
        if 'precision' in metrics:
            results['precision'] = precision_score(y_test, y_pred_test, average='micro')
        if 'recall' in metrics:
            results['recall'] = recall_score(y_test, y_pred_test, average='micro')
        if 'f1_score' in metrics:
            results['f1_score'] = f1_score(y_test, y_pred_test, average='micro')
        if 'roc_auc' in metrics:
            results['roc_auc'] = roc_auc_score(y_test_oh, f_proba_test, multi_class='ovo')

        # Plot metrics
        if 'roc_curve' in metrics:
            logger.warning("ROC Curve plot not available for multi-class classification.")
        if 'pr_curve' in metrics:
            logger.warning("PR Curve plot not available for multi-class classification.")
        if 'confusion_matrix' in metrics:
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)

    return results


def regressor_scores(x_train: Optional[np.ndarray], y_train: Optional[np.ndarray],
                     x_test: np.ndarray, y_test: np.ndarray, rgr: RegressorMixin,
                     metrics: Optional[Union[str, List[str]]] = 'r2_score') -> Dict[str, float]:

    if isinstance(metrics, str):
        metrics = [metrics]
    elif metrics is None:
        metrics = DEFAULT_REGRESSION_METRICS

    missing_metrics = list(filter(lambda m: m not in REGRESSION_METRICS, metrics))
    if len(missing_metrics) > 0:
        raise ValueError("Can't compute following metrics: '{}'".format("', '".join(missing_metrics)))

    try:
        check_is_fitted(rgr)
    except NotFittedError:
        if x_train is None or y_train is None:
            raise ValueError("'x_train' and 'y_train' must be given if the classifier has not been trained")
        rgr.fit(x_train, y_train)

    f_test = rgr.predict(x_test)

    if len(metrics) == 0:
        raise ValueError("Given empty array of metrics")

    results = dict()

    if 'mean_absolute_error' in metrics:
        results['mean_absolute_error'] = mean_absolute_error(y_test, f_test)
    if 'mean_squared_error' in metrics:
        results['mean_squared_error'] = mean_squared_error(y_test, f_test)
    if 'r2_score' in metrics:
        results['r2_score'] = r2_score(y_test, f_test)

    return results


def logistic_regression_r2(df: pd.DataFrame, y_label: str, x_labels: List[str],
                           max_sample_size: int = 10_000, **kwargs) -> Union[None, float]:
    dp = kwargs.get('vf')
    if dp is None:
        dp = MetaExtractor.extract(df=df)
    categorical, continuous = dp.get_categorical_and_continuous()
    if y_label not in [v.name for v in categorical]:
        return None
    if len(x_labels) == 0:
        return None

    df = df[x_labels + [y_label]].dropna()
    df = df.sample(min(max_sample_size, len(df)))

    if df[y_label].nunique() < 2:
        return None

    df_pre = ModellingPreprocessor.preprocess(df, target=y_label, dp=dp)
    x_labels_pre = list(filter(lambda v: v != y_label, df_pre.columns))

    x_array = df_pre[x_labels_pre].to_numpy()
    y_array = df[y_label].to_numpy()

    rg = LogisticRegression()
    rg.fit(x_array, y_array)

    labels = df[y_label].map({c: n for n, c in enumerate(rg.classes_)}).to_numpy()
    oh_labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))

    lp = rg.predict_log_proba(x_array)
    llf = np.sum(oh_labels * lp)

    rg = LogisticRegression()
    rg.fit(np.ones_like(y_array).reshape(-1, 1), y_array)

    lp = rg.predict_log_proba(x_array)
    llnull = np.sum(oh_labels * lp)

    psuedo_r2 = 1 - (llf / llnull)

    return psuedo_r2


def plot_metrics(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                 clf: ClassifierMixin, metrics: Optional[Union[str, List[str]]] = None,
                 axes: Optional[Union[plt.Axes, List[plt.Axes]]] = None, name: str = None):

    if isinstance(metrics, str):
        metrics = [metrics]

    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_PLOT_METRICS
    else:
        missing_metrics = list(filter(lambda m: m not in CLASSIFICATION_PLOT_METRICS, metrics))
        if len(missing_metrics) > 0:
            raise ValueError("Can't compute following metrics: '{}'".format("', '".join(missing_metrics)))

    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif axes is None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
        axes = [axes] if len(metrics) == 1 else axes

    if len(metrics) != len(axes):
        print('metrics', metrics)
        print('axes', axes)
        raise ValueError("Metrics and Axes lengths must be equal")
    axes_dict = {metric: ax for metric, ax in zip(metrics, axes)}

    metrics_to_compute = metrics.copy()
    if 'roc_curve' in metrics:
        metrics_to_compute.extend(['roc_auc'])
    if 'pr_curve' in metrics:
        metrics_to_compute.extend(['f1_score'])
    scores = classifier_scores(x_train, y_train, x_test, y_test, clf, metrics_to_compute)

    if 'roc_curve' in metrics:
        metric = 'roc_curve'
        ax = axes_dict[metric] if metric in axes_dict else plt.axes()

        roc = scores['roc_curve']
        tpr, fpr, _ = roc
        label = f"{name} - " if name is not None else ""
        label += f"AUC={scores['roc_auc']:.3f}"
        ax.plot(tpr, fpr, label=label)
        ax.legend(loc='lower right')
        ax.set_title("ROC Curve")
        ax.set_xlabel("TPR")
        ax.set_ylabel("FPR")

    if 'pr_curve' in metrics:
        metric = 'pr_curve'
        ax = axes_dict[metric] if metric in axes_dict else plt.axes()

        pr = scores['pr_curve']
        prec, rec, _ = pr
        label = f"{name} - " if name is not None else ""
        label += f"F1={scores['f1_score']:.3f}"
        ax.plot(rec, prec, label=label)
        ax.legend(loc='lower right')
        ax.set_title("PR Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    if 'confusion_matrix' in metrics:
        metric = 'confusion_matrix'
        ax = axes_dict[metric] if metric in axes_dict else plt.axes()

        cm = scores['confusion_matrix']
        cm_norm = cm / np.sum(cm)

        cm_annot = _get_cm_annot(cm, cm_norm)
        sns.heatmap(cm_norm, annot=cm_annot, fmt='', vmin=0, vmax=1, annot_kws={"size": 14}, cbar=False, ax=ax)

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")


def _get_cm_annot(cm: np.array, cm_norm: np.array = None) -> List[List[str]]:
    """Compute confusion matrix annotations."""
    if cm_norm is None:
        cm_norm = cm / np.sum(cm)

    cm_annot = []
    for i in range(len(cm)):
        row = []
        for j in range(len(cm[i])):
            row.append(f"{cm[i, j]}\n({cm_norm[i, j] * 100:.2f}%)")
        cm_annot.append(row)

    return cm_annot


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
