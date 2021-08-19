from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..check import ColumnCheck
from .utils import (ConfidenceInterval, MetricStatisticsResult,
                    bootstrap_binned_statistic, bootstrap_interval,
                    bootstrap_pvalue, bootstrap_statistic, permutation_test)


class _Metric(ABC):
    name: Optional[str] = None
    tags: Sequence[str] = []

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.name}"


class OneColumnMetric(_Metric):

    def __init__(self, check: ColumnCheck = None):
        if check is None:
            self.check = ColumnCheck()
        else:
            self.check = check

    @abstractclassmethod
    def check_column_types(cls, check: ColumnCheck, sr: pd.Series) -> bool:
        ...

    @abstractmethod
    def _compute_metric(self, sr: pd.Series):
        ...

    def __call__(self, sr: pd.Series):
        if not self.check_column_types(self.check, sr):
            return None
        return self._compute_metric(sr)


class TwoColumnMetric(_Metric):
    def __init__(self, check: ColumnCheck = None):
        if check is None:
            self.check = ColumnCheck()
        else:
            self.check = check

    @abstractclassmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        ...

    @abstractmethod
    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        ...

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series):
        if not self.check_column_types(self.check, sr_a, sr_b):
            return None
        return self._compute_metric(sr_a, sr_b)


class ModellingMetric(_Metric):

    @abstractmethod
    def __call__(self,
                 y_true: np.ndarray,
                 y_pred: Optional[np.ndarray] = None) -> Union[float, None]:
        pass


class ClassificationMetric(ModellingMetric):
    tags = ["modelling", "classification"]
    plot = False

    def __init__(self, multiclass: bool = False):
        self.multiclass = multiclass

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> float:
        raise NotImplementedError


class ClassificationPlotMetric(ModellingMetric):
    tags = ["modelling", "classification", "plot"]
    plot = True

    def __init__(self, multiclass: bool = False):
        self.multiclass = multiclass

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Any:
        raise NotImplementedError


class RegressionMetric(ModellingMetric):
    tags = ["modelling", "regression"]

    def __init__(self):
        # Contains nothing atm but matches other two Modelling metrics.
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> float:
        raise NotImplementedError


class DataFrameMetric(_Metric):

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> Union[int, float, None]:
        pass


class TwoDataFrameMetric(_Metric):

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[int, float, None]:
        pass


class MetricStatistics(TwoColumnMetric):
    """
    Base class for computing metrics statistics that compare samples from two distributions.

    Args:
        check: ColumnCheck object
        compute_p_val: If p-value should be computed while computing the metric
        compute_interval: If confidence interval should be computed while computing the metric
        confidence_level: Confidence level for computing confidence interval
        bootstrap_mode: If the metric computation is in the bootstrap mode,
                        i,e. in the process of computing confidence interval or p-value
    """
    def __init__(self,
                 check: ColumnCheck = None,
                 compute_p_val: bool = True,
                 compute_interval: bool = True,
                 confidence_level: float = 0.95,
                 bootstrap_mode: bool = False):
        super().__init__(check)
        self.compute_p_val: bool = compute_p_val
        self.compute_interval: bool = compute_interval
        self.confidence_level: float = confidence_level
        self.bootstrap_mode: bool = bootstrap_mode

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series):
        """
        Calculate the distance or correlation between two distributions.

        The MetricStatistics class will be called again and again for the same set of columns
        for computing confidence interval, p-val; we don't want to do perform column check
        corresponding to the metrics on the same set of columns again and again.

        Returns:
            MetricStatisticsResult object contain metric value, p-value and the confidence interval
        """
        if not self.bootstrap_mode and not self.check_column_types(self.check, sr_a, sr_b):
            return None

        return self._compute_metric(sr_a, sr_b)

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        x_dtype = str(check.infer_dtype(sr_a).dtype)
        y_dtype = str(check.infer_dtype(sr_b).dtype)

        return x_dtype == y_dtype

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """The child class will implement this method"""
        pass

    def _compute_p_value(self,
                         sr_a: pd.Series,
                         sr_b: pd.Series,
                         metric_value: float) -> float:
        """Return a p-value for this metric using a permutation test. The null hypothesis
        is that both data samples are from the same distribution."""
        return permutation_test(sr_a, sr_b, lambda x, y: self._metrics_call(x, y))

    def _compute_interval(self,
                          sr_a: pd.Series,
                          sr_b: pd.Series,
                          metric_value: float) -> ConfidenceInterval:
        """Return a frequentist confidence interval for this metric obtained, via bootstrap resampling"""
        samples = bootstrap_statistic((sr_a, sr_b), self._metrics_call)
        return bootstrap_interval(metric_value, samples, self.confidence_level)

    def _metrics_call(self, x, y) -> float:
        cls = type(self)
        obj = cls(compute_p_val=False, compute_interval=False, bootstrap_mode=True)
        return obj(pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)).metric_value


class BinnedMetricStatistics(MetricStatistics):
    """
    Base class for computing metrics statistics that compare counts from two binned distributions
    that have identical binning.

    Args:
        bins: Optional; If given, this must be an iterable of bin edges for x and y,
                i.e the output of np.histogram_bin_edges. If None, then it is assumed
                that the data represent counts of nominal categories, with no meaningful
                distance between bins.
    """
    def __init__(self,
                 check: ColumnCheck = None,
                 compute_p_val: bool = True,
                 compute_interval: bool = True,
                 confidence_level: float = 0.95,
                 bootstrap_mode: bool = False,
                 bins: Optional[Sequence[Any]] = None):
        super().__init__(check, compute_p_val, compute_interval, confidence_level, bootstrap_mode)
        self.bins = bins

    def _compute_p_value(self,
                         sr_a: pd.Series,
                         sr_b: pd.Series,
                         metric_value: float) -> float:
        """Return a two-sided p-value for this metric using a bootstrapped distribution
        of the null hypothesis."""
        ts_distribution = bootstrap_binned_statistic((sr_a, sr_b), self._metrics_call, n_samples=1000)
        return bootstrap_pvalue(metric_value, ts_distribution)

    def _compute_interval(self,
                          sr_a: pd.Series,
                          sr_b: pd.Series,
                          metric_value: float) -> ConfidenceInterval:
        """Compute the frequentist confidence interval for this metric obtained via bootstrap resampling"""
        samples = bootstrap_binned_statistic((sr_a, sr_b), self._metrics_call, n_samples=1000)
        return bootstrap_interval(metric_value, samples, self.confidence_level)
