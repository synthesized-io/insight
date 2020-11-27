"""
Collection of Metrics that measure the distance, or similarity, between two datasets.
"""
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
from abc import abstractmethod

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, binom_test, beta, norm
import pyemd


@dataclass
class ConfidenceInterval():
    value: Tuple[float, float]
    level: float


@dataclass
class DistanceResult():
    distance: float
    p_value: Optional[float] = None
    interval: Optional[ConfidenceInterval] = None


def bootstrap_interval(data, statistic: Callable[..., float], cl: float = 0.95,
                       n_samples: int = 1000, sample_size=None) -> Tuple[float, float]:
    """
    Compute the confidence interval of a statistic estimate using the bootstrap method.

    Args:
        data: Data for which to compute the statistic.
        statistic: Function that computes the statistic.
        args: Extra arguments to statistic
        n_samples: Optional; Number of bootstrap samples to perform.
        cl: Optional; Confidence level of the interval.

    Returns:
        The confidence interval.
    """
    if sample_size is None:
        sample_size = max((len(x) for x in data))

    def get_sample_idx(x):
        return np.random.randint(0, len(x), min(len(x), sample_size))

    statistic_samples = np.empty(n_samples)
    for i in range(n_samples):
        sample_idxs = [get_sample_idx(x) for x in data]
        statistic_samples[i] = statistic(*[x[idx] for x, idx in zip(data, sample_idxs)])

    percentiles = 100 * (1 - cl) / 2, 100 * (1 - (1 - cl) / 2)
    return np.percentile(statistic_samples, percentiles).tolist()


def binominal_proportion_interval(p: float, n: int, cl=0.95, method: str = 'clopper-pearson') -> ConfidenceInterval:
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
        low = beta.ppf(alpha / 2, k, n-k+1)
        high = beta.ppf(1 - alpha / 2, k + 1, n - k)

    elif method == 'agresti-coull':
        n_ = n + z**2
        p_ = 1/n_ * (k + z**2 / 2)
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


def permutation_test(x: np.ndarray, y: np.ndarray, t: Callable[[np.ndarray, np.ndarray], float],
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
        n_per: number of permutations.

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

class DistanceMetric():
    """
    Base class for distance metrics that compare two distributions.

    Subclasses must implement a distance method.

    Attributes:
        bootstrappable: bootstrap resampling can be performed for this metric.
    """
    bootstrap_method: Optional[Callable[..., Tuple[float, float]]] = bootstrap_interval

    def __call__(self, x: pd.Series, y: pd.Series, p_value: bool = False, interval: bool = False) -> DistanceResult:
        """
        Calculate the distance between two distributions.

        Args:
            x: First distribution
            y: Second distribution
            p_value: If True, a p value is calculated. By default this uses a permutation test unless the derived class
            overrides the DistanceMetric.p_value method,
            interval: If True, a 95% confidence interval is calculated using the bootstrap method.

        Returns:
            The calculated result.

        Raises:
            TypeError: interval is True but DistanceMetric.bootstrappable is False.
        """
        result = DistanceResult(self.distance(x, y))
        if p_value:
            result.p_value = self.p_value(x, y)
        if interval:
            result.interval = self.interval(x, y)
        return result

    @abstractmethod
    def distance(self, x: pd.Series, y: pd.Series, **kwargs) -> float:
        """
        Derived classes must implement this.
        """
        ...

    def p_value(self, x: pd.Series, y: pd.Series, **kwargs) -> float:
        """
        Return a p-value for this metric using a permutation test. The null hypothesis
        is that both data samples are from the same distribution.

        Args:
            x (pd.Series): First distribution.
            y (pd.Series): Second distribution.

        Returns:
            The p-value under the null hypothesis.
        """
        return permutation_test(x, y, lambda x, y: self.distance(x, y, **kwargs))

    def interval(self, x: pd.Series, y: pd.Series, cl: float = 0.95, **kwargs) -> ConfidenceInterval:
        """
        Return a frequentist confidence interval for this metric obtained, via bootstrap resampling.

        Args:
            cl: The confidence level of the interval, i.e the fraction of intervals
                   that will contain the true distance estimate.

            **kwargs: Optional keyword arguments to synthesized.insight.metrics.distance.bootstrap_interval.

        Returns:
            The confidence interval.
        """
        if not self.bootstrap_method:
            raise TypeError("Unable to perform bootstrap resampling of this metric.")
        else:
            interval = bootstrap_interval((x, y), lambda x, y: self.distance(x, y, **kwargs), cl=cl)
            return ConfidenceInterval(interval, cl)


class BinomialDistance(DistanceMetric):
    """
    Difference distance between two binomal data samples.
    """
    def distance(self, x: pd.Series, y: pd.Series, **kwargs) -> float:
        """
        Calculate the difference distance, i.e p_x - p_y, where p_x
        is the probability of success in sample x and p_y is the
        probablity of success in sample y.

        Data is assumed to be a series of 1, 0 (success, failure) Bernoulli
        random variates.

        Args:
            x: First binomial data sample.
            y: Second binomial data sample.

        Returns:
            Difference between p_x and p_y.
        """
        return x.mean() - y.mean()

    def p_value(self, x: pd.Series, y: pd.Series, **kwargs) -> float:
        """
        Calculate a p-value for the null hypothesis that the
        probability of success is p_y.

        Args:
            x (pd.Series): First binomial data sample.
            y (pd.Series): Second binomial data sample.

        Returns:
            The p-value under the null hypothesis.
        """
        p_obs = x.mean()
        p_null = y.mean()
        n = len(x)
        return binominal_proportion_p_value(p_obs, p_null, n)

    def interval(self, x: pd.Series, y: pd.Series, cl: Optional[float] = 0.95, **kwargs) -> ConfidenceInterval:
        """
        Calculate a confidence interval for this distance metric.

        Args:
            x: First binomial data sample.
            y: Second binomial data sample.
            cl: Optional; The confidence level of the interval, i.e the fraction of intervals
                that will contain the true distance estimate.

        Returns:
            The confidence interval.
        """
        p = x.mean()
        n = len(x)
        interval = binominal_proportion_interval(p, n, cl, **kwargs)
        interval.value = interval.value[0] - y.mean(), interval.value[1] - y.mean()
        return interval


class KolmogorovSmirnovDistance(DistanceMetric):
    """
    Kolmogorov-Smirnov (KS) distance between two data samples.
    """
    def distance(self, x: pd.Series, y: pd.Series, **kwargs) -> float:
        """
        Calculate the KS distance.

        Args:
            x: First data sample.
            y: Second data sample.
            **kwargs: Optional keyword arguments to scipy.stats.ks_2samp.

        Returns:
            The KS distance.
        """
        return ks_2samp(x, y, **kwargs)[0]


class EarthMoversDistanceBinned(DistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, between two histograms.
    """
    bootstrap_method = None

    def distance(self, x: pd.Series, y: pd.Series, bins: Optional[Tuple] = None, **kwargs) -> float:
        """
        Calculate the EMD between two 1d histograms.

        Histograms must have an equal number of bins. They are not required to be normalised,
        and distances between bins are measured using a Euclidean metric.

        Args:
            x: A series representing histogram counts.
            y: A series representing histogram counts.
            bins: Optional; If given, this must be a tuple of bin edges for x and y,
                i.e the output of np.histogram_bin_edges. If None, then it is assumed
                that the data represent counts of nominal categories, with no meaningful
                distance between bins.
            kwargs: optional keyword arguments to pyemd.emd.

        Returns:
            The earth mover's distance.

        Raises:
            ValueError: x and y do not have the same number of bins.
        """
        if x.shape != y.shape:
            raise ValueError("x and y must have the same number of bins")

        if bins is None:
            # if bins not given, histograms are assumed to be counts of nominal categories,
            # and therefore distances betwen bins are meaningless. Set to all distances to
            # unity to model this.
            distance_metric = 1 - np.eye(len(x))
        else:
            # otherwise, use pair-wise euclidean distances between bin centers for scale data
            bin_centers = [b[:-1] + np.diff(b) / 2. for b in bins]
            mgrid = np.meshgrid(*bin_centers)
            distance_metric = np.abs(mgrid[0] - mgrid[1]).astype(np.float64)

        # normalise counts for consistency with scipy.stats.wasserstein
        x = x / x.sum()
        y = y / y.sum()

        distance = pyemd.emd(x.astype(np.float64), y.astype(np.float64), distance_metric)
        return distance


class EarthMoversDistance(DistanceMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, between samples from two distributions.
    """
    def distance(self, x: pd.Series, y: pd.Series, **kwargs) -> float:
        """
        Calculate the EMD between two 1d histograms.

        Histograms do not have to be normalised, and distances between bins
        are measured using a Euclidean metric.

        Args:
            x: A series representing histogram counts.
            y: A series representing histogram counts.
            kwargs: optional keyword arguments to pyemd.emd_samples.

        Returns:
            The earth mover's distance.
        """
        return pyemd.emd_samples(x, y, **kwargs)


class HellingerDistance(DistanceMetric):

    def __init__(self, x: pd.Series, y: pd.Series, bins = 'auto'):
        self.x = x
        self.y = y
        self.bins = bins

    @property
    def distance(self) -> float:
        x_hist, bins = np.histogram(self.x, bins=self.bins, density=True)
        y_hist, bins = np.histogram(self.y, bins=bins, density=True)
        return 1/np.sqrt(2) * np.sqrt(np.sum((np.sqrt(x_hist) - np.sqrt(y_hist))**2))
