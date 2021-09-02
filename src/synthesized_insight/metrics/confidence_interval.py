from typing import NamedTuple, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from .base import OneColumnMetric, TwoColumnMetric
from .utils import bootstrap_binned_statistic, bootstrap_statistic


class ConfidenceInterval(NamedTuple):
    limits: Tuple
    level: float


def compute_bootstrap_interval(metric_cls_obj: Union[OneColumnMetric, TwoColumnMetric],
                               sr_a: pd.Series,
                               sr_b: pd.Series = None,
                               confidence_level: float = 0.95,
                               binned: bool = False) -> ConfidenceInterval:
    """Return a frequentist confidence interval for this metric obtained, via bootstrap resampling

    Args:
        sr_a: Value of one column
        sr_b: Value of another column
        metric_cls_obj: Instantiated object of a metric class
        confidence_level: Level on which confidence interval is computed
        binned: True in case of binning

    Returns:
        The confidence interval.
    """

    if isinstance(metric_cls_obj, OneColumnMetric):
        metric_value = metric_cls_obj(sr_a)
        one_col_metric: OneColumnMetric = metric_cls_obj  # Need explicit casting because of mypy bug/issue (#2608)
        samples = bootstrap_statistic((sr_a,), lambda x: one_col_metric(x))
    else:
        metric_value = metric_cls_obj(sr_a, cast(pd.Series, sr_b))
        two_col_metric: TwoColumnMetric = metric_cls_obj  # Need explicit casting because of mypy bug/issue (#2608)
        if not binned:
            samples = bootstrap_statistic((sr_a, cast(pd.Series, sr_b)), lambda x, y: two_col_metric(x, y))
        else:
            samples = bootstrap_binned_statistic((sr_a, cast(pd.Series, sr_b)), lambda x, y: two_col_metric(x, y))

    percentiles = 100 * (1 - confidence_level) / 2, 100 * (1 - (1 - confidence_level) / 2)
    d1, d2 = np.percentile(samples, percentiles)
    return ConfidenceInterval(limits=(2 * metric_value - d2, 2 * metric_value - d1), level=confidence_level)


def binominal_proportion_interval(success_prop: float,
                                  num_samples: int,
                                  method: str = 'clopper-pearson',
                                  confidence_level: float = 0.95) -> ConfidenceInterval:
    """
    Calculate an approximate confidence interval for a binomial proportion of a sample.
    Should only be used for binomial distribution

    Args:
        success_prop: Proportion of sucesses.
        num_samples: Sample size.
        method: Optional; The approximation method used to calculate the interval.
            One of 'normal', 'clopper-pearson', 'agresti-coull'.
        confidence_level: Level on which confidence interval is computed

    Returns:
        A ConfidenceInterval containing the interval and confidence level.
    """

    k = num_samples * success_prop
    alpha = 1 - confidence_level
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

    return ConfidenceInterval(limits=(low, high), level=confidence_level)


def compute_binomial_interval(sr_a: pd.Series,
                              sr_b: pd.Series,
                              method: str = 'clopper-pearson',
                              confidence_level: float = 0.95) -> ConfidenceInterval:
    """
    Calculate a confidence interval for this distance metric.
    Args:
        sr_a: value of a binary variable
        sr_b: value of a binary variable
        method: Optional; default is 'clopper-pearson'
        confidence_level: Level on which confidence interval is computed

    Returns:
        The confidence interval.
    """
    p = sr_a.mean()
    n = len(sr_a)
    interval = binominal_proportion_interval(p, n, method, confidence_level)
    cinterval = ConfidenceInterval((interval.limits[0] - sr_b.mean(), interval.limits[1] - sr_b.mean()), interval.level)
    return cinterval
