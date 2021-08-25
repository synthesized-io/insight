from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from .base import TwoColumnMetric, TwoColumnMetricTest
from .statistical_tests import BinomialDistanceTest
from .utils import bootstrap_binned_statistic, bootstrap_statistic


class ConfidenceInterval(NamedTuple):
    limits: Tuple
    level: float


class BootstrapConfidenceInterval:
    def __init__(self,
                 metric_cls_obj: TwoColumnMetric,
                 confidence_level: float = 0.95,
                 binned: bool = False):
        self.metric_cls_obj = metric_cls_obj
        self.confidence_level = confidence_level
        self.binned = binned

    def bootstrap_interval(self,
                           obs: float,
                           bootstrap_samples: pd.Series):
        """
        Calculate the basic bootstrap confidence interval for a statistic
        from a bootstrapped distribution.

        Args:
            obs: The observed statistic value on the sample data.
            bootstrap_samples: Bootstrap samples of the metric.

        Returns:
            The confidence interval.
        """
        percentiles = 100 * (1 - self.confidence_level) / 2, 100 * (1 - (1 - self.confidence_level) / 2)
        d1, d2 = np.percentile(bootstrap_samples, percentiles)
        return ConfidenceInterval(limits=(2 * obs - d2, 2 * obs - d1), level=self.confidence_level)

    def _compute_interval(self,
                          sr_a: pd.Series,
                          sr_b: pd.Series,
                          metric_value: float) -> ConfidenceInterval:
        """Return a frequentist confidence interval for this metric obtained, via bootstrap resampling"""
        if not self.binned:
            samples = bootstrap_statistic((sr_a, sr_b), lambda x, y: self.metric_cls_obj(x, y))
        else:
            samples = bootstrap_binned_statistic((sr_a, sr_b), lambda x, y: self.metric_cls_obj(x, y))
        return self.bootstrap_interval(metric_value, samples)

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series):
        if self.metric_cls_obj is None:
            raise ValueError("Can't perform permutation test without metric class info")

        metric_value = self.metric_cls_obj(sr_a, sr_b)
        return self._compute_interval(sr_a, sr_b, metric_value)


class BinomialInterval:
    def __init__(self,
                 metric_cls_obj: TwoColumnMetricTest,
                 confidence_level: float = 0.95):
        self.metric_cls_obj = metric_cls_obj
        self.confidence_level = confidence_level

    def binominal_proportion_interval(self,
                                      success_prop: float,
                                      num_samples: int,
                                      method: str = 'clopper-pearson') -> ConfidenceInterval:
        """
        Calculate an approximate confidence interval for a binomial proportion of a sample.
        Should only be used for binomial distribution

        Args:
            success_prop: Proportion of sucesses.
            num_samples: Sample size.
            method: Optional; The approximation method used to calculate the interval.
                One of 'normal', 'clopper-pearson', 'agresti-coull'.

        Returns:
            A ConfidenceInterval containing the interval and confidence level.
        """

        k = num_samples * success_prop
        alpha = 1 - self.confidence_level
        z = norm.ppf(1 - alpha / 2)

        if method == 'normal':
            low = success_prop - z * np.sqrt(success_prop * (1 - success_prop) / num_samples)
            high = success_prop + z * np.sqrt(success_prop * (1 - success_prop) / num_samples)

        elif method == 'clopper-pearson':
            low = beta.ppf(alpha / 2, k, num_samples - k + 1)
            high = beta.ppf(1 - alpha / 2, k + 1, num_samples - k)

        elif method == 'agresti-coull':
            n_ = num_samples + z**2
            p_ = 1 / n_ * (k + z**2 / 2)
            low = p_ - z * np.sqrt(p_ * (1 - p_) / n_)
            high = p_ + z * np.sqrt(p_ * (1 - p_) / n_)

        else:
            raise ValueError("'method' argument must be one of 'normal', 'clopper-pearson', 'agresti-coull'.")

        return ConfidenceInterval(limits=(low, high), level=self.confidence_level)

    def _compute_interval(self,
                          sr_a: pd.Series,
                          sr_b: pd.Series,
                          method: str = 'clopper-pearson') -> ConfidenceInterval:
        """
        Calculate a confidence interval for this distance metric.
        Args:
            sr_a: value of a binary variable
            sr_b: value of a binary variable
            metric_value: Metric value used to compute confidence interval
            method: Optional; default is 'clopper-pearson'
        Returns:
            The confidence interval.
        """
        p = sr_a.mean()
        n = len(sr_a)
        interval = self.binominal_proportion_interval(p, n, method)
        cinterval = ConfidenceInterval((interval.limits[0] - sr_b.mean(), interval.limits[1] - sr_b.mean()), interval.level)
        return cinterval

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, method: str = 'clopper-pearson') -> ConfidenceInterval:
        if self.metric_cls_obj is None:
            raise ValueError("Can't perform permutation test without metric class info")

        if not isinstance(self.metric_cls_obj, BinomialDistanceTest):
            raise ValueError("Can't create binomial interval on non-binomial distribution")

        return self._compute_interval(sr_a, sr_b, method)
