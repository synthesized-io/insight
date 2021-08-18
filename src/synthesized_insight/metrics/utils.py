from typing import Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import math
from scipy.stats import beta, binom_test, norm

from ..check import ColumnCheck


class DistrType(Enum):
    """Indicates the type distribution of data in a series."""

    Continuous = "continuous"
    Binary = "binary"
    Categorical = "categorical"
    Datetime = "datetime"

    def is_continuous(self):
        return self == DistrType.Continuous

    def is_binary(self):
        return self == DistrType.Binary

    def is_categorical(self):
        return self == DistrType.Categorical

    def is_datetime(self):
        return self == DistrType.Datetime


def infer_distr_type(column: pd.Series,
                     ctl_mult: float = 2.5,
                     min_num_unique: int = 10) -> DistrType:
    """Infers whether the data in a column or series is datetime, continuous, categorical or binary.
    Args:
        column (pd.Series):
            The column from the data or data series to consider.
        ctl_mult (float, optional):
            Categorical threshold log multiplier. Defaults to 2.5.
        min_num_unique (int, optional):
            Minimum number of unique values for the data to be continuous. Defaults to 10.
    Returns:
        DistrType:
            The output is an enum representing the type of distribution.
    Examples:
        >>> col_type = infer_distr_type(range(1000))
        >>> col_type.is_continuous()
        True
        >>> col_type.is_binary()
        False
    """

    check = ColumnCheck()
    col = check.infer_dtype(column)

    unique = col.unique()
    n_unique = len(unique)
    n_rows = len(col)
    dtype = col.dtype

    if n_unique == 2:
        return DistrType.Binary

    elif dtype == "float64":
        return DistrType.Continuous

    elif dtype == "datetime64[ns]":
        return DistrType.Datetime

    elif n_unique > max(min_num_unique, ctl_mult * np.log(n_rows)) and dtype in ["float64", "int64"]:
        return DistrType.Continuous

    else:
        return DistrType.Categorical


def zipped_hist(
    data: Tuple[pd.Series, ...],
    bin_edges: Optional[np.ndarray] = None,
    normalize: bool = True,
    ret_bins: bool = False,
    distr_type: Optional[str] = None,
) -> Union[Tuple[pd.Series, ...], Tuple[Tuple[pd.Series, ...], Optional[np.ndarray]]]:
    """Bins a tuple of series' and returns the aligned histograms.
    Args:
        data (Tuple[pd.Series, ...]):
            A tuple consisting of the series' to be binned. All series' must have the same dtype.
        bin_edges (Optional[np.ndarray], optional):
            Bin edges to bin continuous data by. Defaults to None.
        normalize (bool, optional):
            Normalize the histograms, turning them into pdfs. Defaults to True.
        ret_bins (bool, optional):
            Returns the bin edges used in the histogram. Defaults to False.
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
    Returns:
        Union[Tuple[np.ndarray, ...], Tuple[Tuple[np.ndarray, ...], Optional[np.ndarray]]]:
            A tuple of np.ndarrays consisting of each histogram for the input data.
            Additionally returns bins if ret_bins is True.
    """

    joint = pd.concat(data)
    is_continuous = distr_type == "continuous" if distr_type is not None else infer_distr_type(joint).is_continuous()

    # Compute histograms of the data, bin if continuous
    if is_continuous:
        # Compute shared bin_edges if not given, and use np.histogram to form histograms
        if bin_edges is None:
            bin_edges = np.histogram_bin_edges(joint, bins="auto")

        hists = [np.histogram(series, bins=bin_edges)[0] for series in data]

        if normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                hists = [np.nan_to_num(hist / hist.sum()) for hist in hists]

    else:
        # For categorical data, form histogram using value counts and align
        space = joint.unique()

        dicts = [sr.value_counts(normalize=normalize) for sr in data]
        hists = [np.array([d.get(val, 0) for val in space]) for d in dicts]

    ps = [pd.Series(hist) for hist in hists]

    if ret_bins:
        return tuple(ps), bin_edges

    return tuple(ps)


@dataclass
class ConfidenceInterval():
    value: Tuple[float, float]
    level: float


@dataclass
class MetricStatisticsResult():
    metric_value: float
    p_value: Optional[float] = None
    interval: Optional[ConfidenceInterval] = None


def affine_mean(sr: pd.Series):
    """function for calculating means of affine values"""
    mean = np.nanmean(sr.values - np.array(0, dtype=sr.dtype))
    return mean + np.array(0, dtype=sr.dtype)


def affine_stddev(sr: pd.Series):
    """function for calculating standard deviations of affine values"""
    d = sr - affine_mean(sr)
    u = d / np.array(1, dtype=d.dtype)
    s = np.sqrt(np.sum(u**2))
    return s * np.array(1, dtype=d.dtype)


def standard_error(sr: pd.Series):
    n = len(sr)
    if n == 0:
        raise ValueError('Series is empty')
    stdv = affine_stddev(sr)
    return stdv / math.sqrt(n)


def bootstrap_interval(obs: float, bootstrap_samples: pd.Series, cl: float = 0.95):
    """
    Calculate the basic bootstrap confidence interval for a statistic
    from a bootstrapped distribution.

    Args:
        obs: The observed statistic value on the sample data.
        bootstrap_samples: Bootstrap samples of the metric.
        cl: Confidence level of the interval.

    Returns:
        The confidence interval.
    """
    percentiles = 100 * (1 - cl) / 2, 100 * (1 - (1 - cl) / 2)
    d1, d2 = np.percentile(bootstrap_samples, percentiles)
    return ConfidenceInterval((2 * obs - d2, 2 * obs - d1), cl)


def bootstrap_pvalue(t_obs: float, t_distribution: pd.Series, alternative: str = 'two-sided'):
    """
    Calculate a p-value using a bootstrapped test statistic distribution

    Args:
        t_obs: Observed value of the test statistic.
        t_distribution: Samples of test statistic distribution under the null hypothesis.
        alternative: Optional; Indicates the alternative hypothesis.
            One of 'two-sided', 'greater' ,'less',

    Returns:
        The p-value under the null hypothesis.
    """
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError("'alternative' argument must be one of 'two-sided', 'greater', 'less'")

    n_samples = len(t_distribution)

    if alternative == 'two-sided':
        p = np.sum(np.abs(t_distribution) >= np.abs(t_obs)) / n_samples

    elif alternative == 'greater':
        p = np.sum(t_distribution >= t_obs) / n_samples

    else:
        p = np.sum(t_distribution < t_obs) / n_samples

    return p


def bootstrap_statistic(data: Union[Tuple[pd.Series], Tuple[pd.Series, pd.Series]],
                        statistic: Union[Callable[[pd.Series, pd.Series], float], Callable[[pd.Series], float]],
                        n_samples: int = 1000, sample_size=None) -> Tuple[float, float]:
    """
    Compute the samples of a statistic estimate using the bootstrap method.

    Args:
        data: Data on which to compute the statistic.
        statistic: Function that computes the statistic.
        n_samples: Optional; Number of bootstrap samples to perform.

    Returns:
        The bootstrap samples.
    """
    if sample_size is None:
        sample_size = max((len(x) for x in data))

    def get_sample_idx(x):
        return np.random.randint(0, len(x), min(len(x), sample_size))

    statistic_samples = np.empty(n_samples)
    for i in range(n_samples):
        sample_idxs = [get_sample_idx(x) for x in data]
        statistic_samples[i] = statistic(*[x[idx] for x, idx in zip(data, sample_idxs)])
    return statistic_samples


def bootstrap_binned_statistic(data: Tuple[pd.Series, pd.Series], statistic: Callable[[pd.Series, pd.Series], float],
                               n_samples: int = 1000) -> Tuple[float, float]:
    """
    Compute the samples of a binned statistic estimate using the bootstrap method.

    Args:
        data: Data for which to compute the statistic.
        statistic: Function that computes the statistic.
        n_samples: Optional; Number of bootstrap samples to perform.

    Returns:
        The bootstrap samples.
    """

    statistic_samples = np.empty(n_samples)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_x = np.nan_to_num(data[0] / data[0].sum())
        p_y = np.nan_to_num(data[1] / data[1].sum())

    n_x = data[0].sum()
    n_y = data[1].sum()

    x_samples = np.random.multinomial(n_x, p_x, size=n_samples)
    y_samples = np.random.multinomial(n_y, p_y, size=n_samples)

    for i in range(n_samples):
        statistic_samples[i] = statistic(x_samples[i], y_samples[i])

    return statistic_samples


def binominal_proportion_interval(p: float, n: int, cl: float = 0.95, method: str = 'clopper-pearson') -> ConfidenceInterval:
    """
    Calculate an approximate confidence interval for a binomial proportion of a sample.

    Args:
        p: Proportion of sucesses.
        n: Sample size.
        cl: Optional; Confidence level of the interval.
        method: Optional; The approximation method used to calculate the interval.
            One of 'normal', 'clopper-pearson', 'agresti-coull'.

    Returns:
        A ConfidenceInterval containing the interval and confidence level.
    """

    k = n * p
    alpha = 1 - cl
    z = norm.ppf(1 - alpha / 2)

    if method == 'normal':
        low = p - z * np.sqrt(p * (1 - p) / n)
        high = p + z * np.sqrt(p * (1 - p) / n)

    elif method == 'clopper-pearson':
        low = beta.ppf(alpha / 2, k, n - k + 1)
        high = beta.ppf(1 - alpha / 2, k + 1, n - k)

    elif method == 'agresti-coull':
        n_ = n + z**2
        p_ = 1 / n_ * (k + z**2 / 2)
        low = p_ - z * np.sqrt(p_ * (1 - p_) / n_)
        high = p_ + z * np.sqrt(p_ * (1 - p_) / n_)

    else:
        raise ValueError("'method' argument must be one of 'normal', 'clopper-pearson', 'agresti-coull'.")

    return ConfidenceInterval((low, high), cl)


def binominal_proportion_p_value(p_obs: float, p_null: float, n: int, alternative: str = 'two-sided') -> float:
    """
    Calculate an exact p-value for an observed binomial proportion of a sample.

    Args:
        p_obs: Observed proportion of successes.
        p_null: Expected proportion of sucesses under null hypothesis.
        n: Sample size.
        alternative: Optional; Indicates the alternative hypothesis.
            One of 'two-sided', 'greater' ,'less',

    Returns:
        The p-value under the null hypothesis.
    """
    k = np.ceil(p_obs * n)
    return binom_test(k, n, p_null, alternative)


def permutation_test(x: pd.Series, y: pd.Series, t: Callable[[pd.Series, pd.Series], float],
                     n_perm: int = 100, alternative: str = 'two-sided') -> float:
    """
    Perform a two sample permutation test.

    Determines the probability of observing t(x, y) or greater under the null hypothesis that x
    and y are from the same distribution.

    Args:
        x: First data sample.
        y: Second data sample.
        t: Callable that returns the test statistic.
        alternative: Optional; Indicates the alternative hypothesis.
            One of 'two-sided', 'greater' ,'less',
        n_perm: number of permutations.

    Returns:
        The p-value of t_obs under the null hypothesis.
    """

    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError("'alternative' argument must be one of 'two-sided', 'greater', 'less'")

    t_obs = t(x, y)
    pooled_data = np.concatenate((x, y))
    t_null = np.empty(n_perm)

    for i in range(n_perm):
        perm = np.random.permutation(pooled_data)
        x_sample = perm[:len(x)]
        y_sample = perm[len(x):]
        t_null[i] = t(x_sample, y_sample)

    if alternative == 'two-sided':
        p = np.sum(np.abs(t_null) >= np.abs(t_obs)) / n_perm

    elif alternative == 'greater':
        p = np.sum(t_null >= t_obs) / n_perm

    else:
        p = np.sum(t_null < t_obs) / n_perm

    return p
