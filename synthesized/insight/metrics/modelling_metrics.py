import logging
from typing import Any, Dict, List, Optional, Type, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.utils.validation import check_is_fitted

from .metrics_base import (ClassificationMetric, ClassificationPlotMetric, DataFrameMetric, RegressionMetric,
                           TwoDataFrameMetric)
from ..modelling import ModellingPreprocessor, check_model_type, preprocess_split_data
from ...metadata import DataFrameMeta, Nominal
from ...metadata.factory import MetaExtractor
from ...model import ContinuousModel, DataFrameModel, DiscreteModel
from ...model.factory import ModelFactory

logger = logging.getLogger(__name__)

MAX_ANALYSIS_SAMPLE_SIZE = 10_000


class PredictiveModellingScore(DataFrameMetric):
    name = "predictive_modelling_score"
    tags = ["modelling"]

    def __init__(
        self, model: str = None, y_label: str = None, x_labels: List[str] = None, sample_size: Optional[int] = None
    ):
        self.model = model or 'Linear'
        self.y_label = y_label
        self.x_labels = x_labels
        self.sample_size = sample_size

    def __call__(self, df: pd.DataFrame, df_model: DataFrameModel = None) -> Union[int, float, None]:
        if len(df.columns) < 2:
            raise ValueError
        y_label = self.y_label or df.columns[-1]
        x_labels = self.x_labels if self.x_labels is not None else [col for col in df.columns if col != y_label]

        score, metric, task = predictive_modelling_score(df, y_label, x_labels, self.model,
                                                         sample_size=self.sample_size)
        return score


class PredictiveModellingComparison(TwoDataFrameMetric):
    name = "predictive_modelling_comparison"
    tags = ["modelling"]

    def __init__(
        self, model: str = None, y_label: str = None, x_labels: List[str] = None, sample_size: Optional[int] = None
    ):
        self.model = model or 'Linear'
        self.y_label = y_label
        self.x_labels = x_labels
        self.sample_size = sample_size

    def __call__(
            self, df_old: pd.DataFrame, df_new: pd.DataFrame, df_model: DataFrameModel = None
    ) -> Union[None, float]:

        if len(df_old.columns) < 2:
            raise ValueError

        y_label = self.y_label or df_old.columns[-1]
        x_labels = self.x_labels if self.x_labels is not None else [col for col in df_old.columns if col != y_label]

        score, synth_score, metric, task = predictive_modelling_comparison(
            df_old, df_new, y_label, x_labels, self.model, sample_size=self.sample_size
        )
        return synth_score / score


class Accuracy(ClassificationMetric):
    name = "accuracy"

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Union[float]:
        assert y_pred is not None
        return accuracy_score(y_true, y_pred)


class Precision(ClassificationMetric):
    name = "precision"

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Union[float]:
        assert y_pred is not None
        if self.multiclass is False:
            return precision_score(y_true, y_pred)
        else:
            return precision_score(y_true, y_pred, average='micro')


class Recall(ClassificationMetric):
    name = "recall"

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Union[float]:
        assert y_pred is not None
        if self.multiclass is False:
            return recall_score(y_true, y_pred)
        else:
            return recall_score(y_true, y_pred, average='micro')


class F1Score(ClassificationMetric):
    name = "f1_score"

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Union[float]:
        assert y_pred is not None
        if self.multiclass is False:
            return f1_score(y_true, y_pred)
        else:
            return f1_score(y_true, y_pred, average='micro')


class ROC_AUC(ClassificationMetric):
    name = "roc_auc"

    def __call__(self, y_true: np.ndarray = None, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Union[float]:
        assert y_pred_proba is not None
        if len(np.unique(y_true)) < 2:
            return 1.0
        if self.multiclass is False:
            return roc_auc_score(y_true, y_pred_proba)
        else:
            return roc_auc_score(y_true, y_pred_proba, multi_class='ovo')


class ROC_Curve(ClassificationPlotMetric):
    name = "roc_curve"

    def __init__(self, multiclass: bool = False):
        self.plot = True
        super(ROC_Curve, self).__init__(multiclass=multiclass)

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Union[float, None]:
        assert y_pred_proba is not None
        if self.multiclass is False:
            return roc_curve(y_true, y_pred_proba)
        else:
            logger.warning("ROC Curve plot not available for multi-class classification.")
            return None


class PR_Curve(ClassificationPlotMetric):
    name = "pr_curve"

    def __init__(self, multiclass: bool = False):
        self.plot = True
        super(PR_Curve, self).__init__(multiclass=multiclass)

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Union[float, None]:
        assert y_pred_proba is not None
        if self.multiclass is False:
            return precision_recall_curve(y_true, y_pred_proba)
        else:
            logger.warning("PR Curve plot not available for multi-class classification.")
            return None


class ConfusionMatrix(ClassificationPlotMetric):
    name = "confusion_matrix"

    def __init__(self, multiclass: bool = False):
        self.plot = True
        super(ConfusionMatrix, self).__init__(multiclass=multiclass)

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Union[int, float, None]:
        assert y_pred is not None
        return confusion_matrix(y_true, y_pred)


class MeanAbsoluteError(RegressionMetric):
    name = "mean_absolute_error"

    def __init__(self):
        super(MeanAbsoluteError, self).__init__()

    def __call__(self, y_true: np.ndarray = None, y_pred: np.ndarray = None, **kwargs) -> Union[float]:
        assert y_true is not None and y_pred is not None
        return mean_absolute_error(y_true, y_pred)


class MeanSquaredError(RegressionMetric):
    name = "mean_squared_error"

    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def __call__(self, y_true: np.ndarray = None, y_pred: np.ndarray = None, **kwargs) -> Union[float]:
        assert y_true is not None and y_pred is not None
        return mean_squared_error(y_true, y_pred)


class R2_Score(RegressionMetric):
    name = "r2_score"

    def __init__(self):
        super(R2_Score, self).__init__()

    def __call__(self, y_true: np.ndarray = None, y_pred: np.ndarray = None, **kwargs) -> Union[float]:
        assert y_true is not None and y_pred is not None
        return r2_score(y_true, y_pred)


REGRESSION_METRICS: Dict[str, Type[RegressionMetric]] = {
    cast(str, m.name): m for m in [R2_Score, MeanSquaredError, MeanAbsoluteError]
}
CLASSIFICATION_METRICS: Dict[str, Type[ClassificationMetric]] = {
    cast(str, m.name): m for m in [Accuracy, Precision, Recall, F1Score, ROC_AUC]
}
CLASSIFICATION_PLOT_METRICS: Dict[str, Type[ClassificationPlotMetric]] = {
    cast(str, m.name): m for m in [ROC_Curve, PR_Curve, ConfusionMatrix]
}


def predictive_modelling_score(
    data: pd.DataFrame,
    y_label: str,
    x_labels: Optional[List[str]],
    model: Union[str, BaseEstimator],
    synth_data: pd.DataFrame = None,
    copy_model: bool = True,
    preprocessor: ModellingPreprocessor = None,
    dp: DataFrameMeta = None,
    models: DataFrameModel = None,
    sample_size: Optional[int] = None
):
    """Calculates the R-squared or ROC AUC score of a given model trained on a given dataset.

    This function will fit a regressor or classifier depending on the datatype of the y_label. All necessary
    preprocessing (e.g standard scaling, one-hot encoding) is done in the function. The input data is automatically
    split into a training and testing set in order to evaluate the model performance.

    Args:
        data (pd.DataFrame): Input dataset.
        y_label (str): Name of the target variable column/response variable to predict.
        x_labels (List[str], optional): Input column names/explanatory variables. Defaults to None, in which case all
            columns in the dataset except y_label will be used as predictors.
        model (Union[str, sklearn.base.BaseEstimator]): One of 'Linear', 'GradientBoosting', 'RandomForrest', 'MLP',
            'LinearSVM', or 'Logistic'. Note that 'Logistic' only applies to categorical response variables.
            Alternatively, a custom model class that inherits from sklearn.base.BaseEstimator can be specified.
        synth_data (pd.DataFrame): Train the model on this synthetic data but evaluate it's performance on the original.

    Returns:
        The score, metric ('r2' or 'roc_auc'), and the task ('regression', 'binary', or 'multinomial')
    """

    score, metric, task = None, None, None

    # Remove rows with response variable as NaN before splitting,
    # because stratified train_test_split will fail if response contains NaNs.
    # Also, NaN response value doesn't help with supervised learning
    response_nan_idxs = data.index[data[y_label].isna()].tolist()
    if response_nan_idxs:
        data = data.drop(labels=response_nan_idxs, axis=0)

    if x_labels is None:
        x_labels = list(filter(lambda c: c != y_label, data.columns))
    else:
        available_columns = list(set(data.columns).intersection(set(x_labels + [y_label])))
        if y_label not in available_columns or len([label for label in x_labels if label in available_columns]) == 0:
            raise ValueError('Response variable not in DataFrame.')
        data = data[available_columns]

    if dp is None:
        dp = MetaExtractor.extract(df=data)
    if models is None:
        models = ModelFactory()(dp)

    x_labels = list(filter(lambda v: v in cast(DataFrameMeta, dp), x_labels))

    if y_label not in dp:
        raise ValueError('Response variable type not handled.')
    elif len(x_labels) == 0:
        raise ValueError('No explanatory variables with acceptable type.')

    # Check predictor
    if isinstance(models[y_label], DiscreteModel):
        estimator = check_model_type(model, copy_model, 'clf')
    elif isinstance(models[y_label], ContinuousModel):
        estimator = check_model_type(model, copy_model, 'rgr')
    else:
        raise ValueError(f"Can't understand y_label '{y_label}' type.")

    # If sample_size is not passed as the parameter of the function,
    # then set it to the minimum of MAX_ANALYSIS_SAMPLE_SIZE and data size
    if sample_size is None:
        sample_size = min(MAX_ANALYSIS_SAMPLE_SIZE, len(data))

    if preprocessor is None:
        preprocessor = ModellingPreprocessor(target=y_label, df_model=models)
        if synth_data is not None:
            # fit data together as preprocessor needs to know all categorical variables
            preprocessor.fit(pd.concat((data, synth_data)))

    df_train_pre, df_test_pre = preprocess_split_data(data, response_variable=y_label, explanatory_variables=x_labels,
                                                      sample_size=sample_size, preprocessor=preprocessor)
    if synth_data is not None:
        synth_data = synth_data[available_columns]
        df_train_pre, _ = preprocess_split_data(synth_data, response_variable=y_label, explanatory_variables=x_labels,
                                                sample_size=sample_size, preprocessor=preprocessor)

    x_labels_pre = list(filter(lambda v: v != y_label, df_train_pre.columns))
    x_train = df_train_pre[x_labels_pre].to_numpy()
    y_train = df_train_pre[y_label].to_numpy()
    x_test = df_test_pre[x_labels_pre].to_numpy()
    y_test = df_test_pre[y_label].to_numpy()

    if isinstance(models[y_label], ContinuousModel):
        metric = 'r2_score'
        task = 'regression'
        scores = regressor_scores(x_train, y_train, x_test, y_test, rgr=estimator, metrics=metric)
        score = scores[metric]

    elif isinstance(models[y_label], DiscreteModel):
        y_val = cast(Nominal, dp[y_label])
        assert y_val.categories is not None and len(y_val.categories)
        num_classes = len(y_val.categories)
        metric = 'roc_auc' if num_classes == 2 else 'macro roc_auc'
        task = 'binary ' if num_classes == 2 else f'multinomial [{num_classes}]'
        scores = classifier_scores(x_train, y_train, x_test, y_test, clf=estimator, metrics='roc_auc')
        score = scores['roc_auc']

    return score, metric, task


def predictive_modelling_comparison(
    data: pd.DataFrame,
    synth_data: pd.DataFrame,
    y_label: str,
    x_labels: List[str],
    model: str,
    sample_size: Optional[int] = None
):
    """Compare the R-squared or ROC AUC score of a model trained on original data and synthetic data, and tested on
    hold-out sample of original data.

    This function will fit a regressor or classifier depending on the datatype of the y_label. All necessary
    preprocessing (e.g standard scaling, one-hot encoding) is done in the function. The input data is automatically
    split into a training and testing set in order to evaluate the model performance.

    Args:
        data (pd.DataFrame): Original data.
        synth_data (pd.DataFrame): Synthetic data.
        y_label (str): Name of the target variable column/response variable to predict.
        x_labels (List[str], optional): Input column names/explanatory variables. Defaults to None, in which case all
            columns in the dataset except y_label will be used as predictors.
        model (Union[str, sklearn.base.BaseEstimator]): One of 'Linear', 'GradientBoosting', 'RandomForrest', 'MLP',
            'LinearSVM', or 'Logistic'. Note that 'Logistic' only applies to categorical response variables.
            Alternatively, a custom model class that inherits from sklearn.base.BaseEstimator can be specified.
        sample_size (int, optional): Size of dataset to use for training data. Defaults to None, in which case the full
            dataset is used to create the training and test sample.

    Returns:
        The score, synthetic score, metric ('r2' or 'roc_auc'), and the task ('regression', 'binary', or 'multinomial')
    """
    score, metric, task = predictive_modelling_score(data, y_label, x_labels, model, sample_size=sample_size)
    synth_score, _, _ = predictive_modelling_score(data, y_label, x_labels, model, synth_data=synth_data, sample_size=sample_size)

    return score, synth_score, metric, task


def classifier_scores_from_df(df_train: Optional[pd.DataFrame], df_test: pd.DataFrame,
                              target: str, clf: ClassifierMixin, metrics: Optional[Union[str, List[str]]] = 'roc_auc',
                              return_predicted: bool = False) -> Dict[str, Any]:
    x_labels = list(filter(lambda v: v != target, df_test.columns))

    if df_train is not None:
        x_train = df_train[x_labels].to_numpy()
        y_train = df_train[target].to_numpy()
    else:
        x_train = y_train = None
    x_test = df_test[x_labels].to_numpy()
    y_test = df_test[target].to_numpy()

    return classifier_scores(x_train, y_train, x_test, y_test, clf=clf, metrics=metrics,
                             return_predicted=return_predicted)


def regressor_scores_from_df(df_train: Optional[pd.DataFrame], df_test: pd.DataFrame,
                             target: str, rgr: RegressorMixin, metrics: Optional[Union[str, List[str]]] = 'r2_score',
                             return_predicted: bool = False) -> Dict[str, Any]:
    x_labels = list(filter(lambda v: v != target, df_test.columns))

    if df_train is not None:
        x_train = df_train[x_labels].to_numpy()
        y_train = df_train[target].to_numpy()
    else:
        x_train = y_train = None
    x_test = df_test[x_labels].to_numpy()
    y_test = df_test[target].to_numpy()

    return regressor_scores(x_train, y_train, x_test, y_test, rgr=rgr, metrics=metrics,
                            return_predicted=return_predicted)


def plot_metrics_from_df(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str, clf: ClassifierMixin,
                         metrics: Optional[Union[str, List[str]]] = None,
                         axes: Optional[Union[plt.Axes, List[plt.Axes]]] = None, name: str = None):
    x_labels = list(filter(lambda v: v != target, df_train.columns))
    x_train = df_train[x_labels].to_numpy()
    y_train = df_train[target].to_numpy()
    x_test = df_test[x_labels].to_numpy()
    y_test = df_test[target].to_numpy()

    plot_metrics(x_train, y_train, x_test, y_test, clf=clf, metrics=metrics, axes=axes, name=name)


def classifier_scores(x_train: np.ndarray, y_train: np.ndarray,
                      x_test: np.ndarray, y_test: np.ndarray,
                      clf: ClassifierMixin, metrics: Optional[Union[str, List[str]]] = 'roc_auc',
                      return_predicted: bool = False) -> Dict[str, Any]:

    all_classification_metrics: Dict[str, Union[Type[ClassificationMetric], Type[ClassificationPlotMetric]]] = dict(
        CLASSIFICATION_METRICS, **CLASSIFICATION_PLOT_METRICS
    )
    if isinstance(metrics, str):
        metrics_dict = {
            metrics: all_classification_metrics[metrics]
        }
    elif isinstance(metrics, list):
        metrics_dict = {metric_name: all_classification_metrics[metric_name] for metric_name in metrics}
    elif metrics is None:
        metrics_dict = {metric_name: CLASSIFICATION_METRICS[metric_name]
                        for metric_name in CLASSIFICATION_METRICS.keys()}
    else:
        raise TypeError("'metrics' type not recognized")

    # Check all metrics are supported
    missing_metrics = list(filter(lambda m: m not in all_classification_metrics.keys(), metrics_dict.keys()))
    if len(missing_metrics) > 0:
        raise ValueError("Can't compute following metrics: '{}'".format("', '".join(missing_metrics)))

    # Single class present in target
    if len(np.unique(y_train)) == 1:
        return {metric_name: 1. for metric_name in metrics_dict.keys()}

    # Check if fitted
    try:
        check_is_fitted(clf)
    except NotFittedError:
        clf.fit(x_train, y_train)

    # Two classes classification
    if len(np.unique(y_train)) == 2:
        multiclass = False
        if hasattr(clf, 'predict_proba'):
            f_proba_test = clf.predict_proba(x_test)
            y_pred_test = np.argmax(f_proba_test, axis=1)
            f_proba_test = f_proba_test.T[1]
        else:
            y_pred_test = f_proba_test = clf.predict(x_test)

    # Multi-class classification
    else:
        multiclass = True
        oh = OneHotEncoder(sparse=False)
        oh.fit(np.concatenate((y_train, y_test)).reshape((-1, 1)))
        y_test_oh = oh.transform(y_test.reshape((-1, 1)))

        if hasattr(clf, 'predict_proba'):
            missing = [v for v in oh.categories_[0] if v not in clf.classes_]
            f_proba_test = clf.predict_proba(x_test)
            missing_idx = sorted([list(oh.categories_[0]).index(v) for v in missing], reverse=True)
            for idx in missing_idx:
                f_proba_test = np.insert(f_proba_test, missing_idx, 0, 1)
            y_pred_test = np.array(oh.categories_[0])[np.argmax(f_proba_test, axis=1)]
        else:
            y_pred_test = clf.predict(x_test)
            f_proba_test = oh.transform(y_pred_test.reshape(-1, 1))

        y_test_oh, f_proba_test = _remove_zero_column(y_test_oh, f_proba_test)
        # Normalise the prediction probabilities so that the probabilities across classes add to 1.
        f_proba_test = _normalise_prob(f_proba_test)

    results: Dict[str, Any] = dict()
    for metric_name, metric in metrics_dict.items():
        # metric() and metric.__call__() are the same, but the second raises lint error
        m_inst = metric(multiclass=multiclass)
        results[metric_name] = m_inst(y_true=y_test, y_pred=y_pred_test,
                                      y_pred_proba=f_proba_test)

    if return_predicted:
        results['predicted_values'] = y_pred_test

    return results


def regressor_scores(x_train: Optional[np.ndarray], y_train: Optional[np.ndarray],
                     x_test: np.ndarray, y_test: np.ndarray,
                     rgr: RegressorMixin, metrics: Optional[Union[str, List[str]]] = 'r2_score',
                     return_predicted: bool = False) -> Dict[str, float]:

    if isinstance(metrics, str):
        metrics_dict: Dict[str, Type[RegressionMetric]] = {metrics: REGRESSION_METRICS[metrics]}
    elif isinstance(metrics, list):
        if len(metrics) == 0:
            raise ValueError("Given empty array of metrics")
        metrics_dict = {metric_name: REGRESSION_METRICS[metric_name] for metric_name in metrics}
    elif metrics is None:
        metrics_dict = {metric_name: REGRESSION_METRICS[metric_name] for metric_name in REGRESSION_METRICS.keys()}
    else:
        raise TypeError("'metrics' type not recognized")

    # Check all metrics are supported
    missing_metrics = list(filter(lambda m: m not in REGRESSION_METRICS.keys(), metrics_dict.keys()))
    if len(missing_metrics) > 0:
        raise ValueError("Can't compute following metrics: '{}'".format("', '".join(missing_metrics)))

    try:
        check_is_fitted(rgr)
    except NotFittedError:
        if x_train is None or y_train is None:
            raise ValueError("'x_train' and 'y_train' must be given if the classifier has not been trained")
        rgr.fit(x_train, y_train)

    f_test = rgr.predict(x_test)

    results: Dict[str, Any] = dict()
    for metric_name, metric in metrics_dict.items():
        results[metric_name] = metric()(y_true=y_test, y_pred=f_test)

    if return_predicted:
        results['predicted_values'] = f_test

    return results


def plot_metrics(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                 clf: ClassifierMixin, metrics: Optional[Union[str, List[str]]] = None,
                 axes: Optional[Union[plt.Axes, List[plt.Axes]]] = None, name: str = None):

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
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        axes = [axes] if len(metrics) == 1 else axes

    if len(metrics) != len(axes):
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

        if name:
            ax.set_title(f"Confusion Matrix - {name}")
        else:
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
    if y1.shape[1] != y2.shape[1]:
        raise ValueError(f"Mismatch of array shapes, {y1.shape} not compatible with {y2.shape}.")

    delete_index = np.where((y1 == 0).all(axis=0))

    y1 = np.delete(y1, delete_index, axis=1)
    y2 = np.delete(y2, delete_index, axis=1)
    return y1, y2


def _normalise_prob(f_proba):
    """Given probabilities across classes, normalise them so that they add to 1 for each row.

    Args:
        f_proba: Input probability array

    Returns:
        f_proba_normalised: Output normalised probability array

    """
    f_proba_normalised = normalize(f_proba, axis=1, norm='l1')
    idxs = np.where(~f_proba_normalised.any(axis=1))[0]
    num_classes = f_proba_normalised.shape[1]
    f_proba_normalised[idxs] = 1 / num_classes

    return f_proba_normalised
