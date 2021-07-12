import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import ClassifierMixin, clone

from ...insight.metrics.modelling_metrics import CLASSIFICATION_PLOT_METRICS, classifier_scores
from ...insight.modelling import ModellingPreprocessor
from ...metadata.factory import MetaExtractor
from ...model import DataFrameModel
from ...model.factory import ModelFactory

logger = logging.getLogger(__name__)


def plot_classification_metrics(df_model: DataFrameModel, target: str, df_train1: pd.DataFrame, df_train2: pd.DataFrame,
                                df_test: pd.DataFrame, clf: ClassifierMixin, names: Optional[Tuple[str, str]] = None):
    """Compare the performance on the same dataset (df_test) of a classifier trained on two different datasets
    .
    Given two training sets, and one test set, plot a comparision of ROC, F1 and Confusion matrix"""

    name1 = names[0] if names else "Train Set 1"
    name2 = names[1] if names else "Train Set 2"

    preprocessor = ModellingPreprocessor(target=target, df_model=df_model)
    preprocessor.fit(pd.concat([df_train1, df_train2, df_test], ignore_index=True))

    _, axs = plt.subplots(2, 2, figsize=(12, 12))
    axes1: List[plt.Axes] = [axs[0, 0], axs[0, 1], axs[1, 0]]
    axes2: List[plt.Axes] = [axs[0, 0], axs[0, 1], axs[1, 1]]

    plot_metrics_from_df(df_train1, df_test, target, clf=clone(clf), preprocessor=preprocessor,
                         axes=axes1, name=name1)
    plot_metrics_from_df(df_train2, df_test, target, clf=clone(clf), preprocessor=preprocessor,
                         axes=axes2, name=name2)


def plot_classification_metrics_test(df_model: DataFrameModel, target: str, df_train: pd.DataFrame,
                                     df_test1: pd.DataFrame, df_test2: pd.DataFrame, clf: ClassifierMixin,
                                     names: Optional[Tuple[str, str]] = None):
    """Given one training set, and two test sets, plot a comparision of ROC, F1 and Confusion matrix"""

    name1 = names[0] if names else "Test Set 1"
    name2 = names[1] if names else "Test Set 2"

    preprocessor = ModellingPreprocessor(target=target, df_model=df_model)
    preprocessor.fit(pd.concat([df_train, df_test1, df_test2], ignore_index=True))

    clf = clone(clf)

    _, axs = plt.subplots(2, 2, figsize=(12, 12))
    axes1: List[plt.Axes] = [axs[0, 0], axs[0, 1], axs[1, 0]]
    axes2: List[plt.Axes] = [axs[0, 0], axs[0, 1], axs[1, 1]]

    plot_metrics_from_df(df_train, df_test1, target, clf=clf, preprocessor=preprocessor,
                         axes=axes1, name=name1)
    plot_metrics_from_df(df_train, df_test2, target, clf=clf, preprocessor=preprocessor,
                         axes=axes2, name=name2)


def plot_metrics_from_df(df_train: pd.DataFrame, df_test: pd.DataFrame,
                         target: str, clf: ClassifierMixin,
                         preprocessor: Optional[ModellingPreprocessor] = None,
                         metrics: Optional[Union[str, List[str]]] = None,
                         axes: Optional[Union[plt.Axes, List[plt.Axes]]] = None, name: str = None):
    """Given a train and test dataframes and a classifier, plot ROC, F1 and Confusion matrix"""

    if preprocessor is None:
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        if len(df_all) > 20_000:
            df_all = df_all.sample(20_000)
        df_meta = MetaExtractor.extract(df_all)
        df_models = ModelFactory()(df_meta)
        preprocessor = ModellingPreprocessor(target=target, df_model=df_models)
        preprocessor.fit(df_all)

    df_train = preprocessor.transform(df_train)
    df_test = preprocessor.transform(df_test)

    x_labels = list(filter(lambda v: v != target, df_train.columns))
    x_train = df_train[x_labels].to_numpy()
    y_train = df_train[target].to_numpy()
    x_test = df_test[x_labels].to_numpy()
    y_test = df_test[target].to_numpy()

    plot_metrics(x_train, y_train, x_test, y_test, clf=clf, metrics=metrics, axes=axes, name=name)


def _check_metrics_axes(
    metrics: Optional[Union[str, List[str]]] = None,
    axes: Optional[Union[plt.Axes, List[plt.Axes]]] = None
) -> Tuple[List[str], Dict[str, plt.Axes]]:

    if isinstance(metrics, str):
        metrics = [metrics]

    if metrics is None:
        metrics = list(CLASSIFICATION_PLOT_METRICS.keys())
    else:
        missing_metrics = list(filter(lambda m: m not in CLASSIFICATION_PLOT_METRICS, metrics))
        if len(missing_metrics) > 0:
            raise ValueError("Can't compute following metrics: '{}'".format("', '".join(missing_metrics)))

    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif axes is None:
        _, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        axes = [axes] if len(metrics) == 1 else axes

    if len(metrics) != len(axes):
        raise ValueError("Metrics and Axes lengths must be equal")
    axes_dict = {metric: ax for metric, ax in zip(metrics, axes)}

    assert all([metric in axes_dict for metric in metrics])
    return metrics, axes_dict


def plot_metrics(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                 clf: ClassifierMixin, metrics: Optional[Union[str, List[str]]] = None,
                 axes: Optional[Union[plt.Axes, List[plt.Axes]]] = None, name: str = None):

    metrics, axes_dict = _check_metrics_axes(metrics, axes)

    metrics_to_compute = metrics.copy()
    if 'roc_curve' in metrics:
        metrics_to_compute.extend(['roc_auc'])
    if 'pr_curve' in metrics:
        metrics_to_compute.extend(['f1_score'])
    scores = classifier_scores(x_train, y_train, x_test, y_test, clf, metrics_to_compute)

    if 'roc_curve' in metrics:
        ax = axes_dict['roc_curve']
        roc = scores['roc_curve']
        tpr, fpr, _ = roc
        label = f"{f'{name} - ' if name is not None else ''}AUC={scores['roc_auc']:.3f}"
        ax.plot(tpr, fpr, label=label)
        ax.legend(loc='lower right')
        ax.set_title("ROC Curve")
        ax.set_xlabel("TPR")
        ax.set_ylabel("FPR")

    if 'pr_curve' in metrics:
        ax = axes_dict['pr_curve']
        pr = scores['pr_curve']
        prec, rec, _ = pr
        label = f"{f'{name} - ' if name is not None else ''}F1={scores['f1_score']:.3f}"
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

        plt_title = "Confusion Matrix"
        if name:
            plt_title += f" - {name}"
        ax.set_title(plt_title)
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
