"""
Collection of Metrics that measure the distance, or similarity, between two datasets.
"""
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
from abc import abstractmethod

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import pyemd


def bootstrap_interval(data, statistic: Callable[..., float], cl: float = 0.95,
                       n_samples: int = 1000, sample_size=None):
    """
    Compute the confidence interval of a statistic estimate using the bootstrap method.

    Arguments:

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


def permutation_test(x: np.ndarray, y: np.ndarray, t: Callable[[np.ndarray, np.ndarray], float],
                     two_sided: bool = True, n_perm: int = 100) -> float:
    """
    Perform a two sample permutation test.

    Determines the probability of observing t(x, y) or greater under the null hypothesis that x
    and y are from the same distribution.

    Arguments:
        x: First data sample.

        y: Second data sample.

        t: Callable that returns the test statistic.

        two_sided: If True, a two-sided p-value is returned, else one-sided.

        n_per: number of permutations.

    Returns:
        The p-value of t_obs under the null hypothesis.
    """

    t_obs = t(x, y)
    pooled_data = np.concatenate((x, y))
    t_null = np.empty(n_perm)

    for i in range(n_perm):
        perm = np.random.permutation(pooled_data)
        x_sample = perm[:len(x)]
        y_sample = perm[len(x):]
        t_null[i] = t(x_sample, y_sample)

    if two_sided:
        return np.sum(np.abs(t_null) >= np.abs(t_obs)) / n_perm

    else:
        return np.sum(t_null >= t_obs) / n_perm


@dataclass
class MetricResult():
    pass


@dataclass
class ConfidenceInterval():
    value: Tuple[float, float]
    level: float


@dataclass
class DistanceResult(MetricResult):
    distance: float
    p_value: Optional[float] = None
    interval: Optional[ConfidenceInterval] = None


class DistanceMetric():
    """
    Base class for distance metrics that compare two distributions.

    Subclasses must implement a distance method.

    Attributes:
        bootstrappable: bootstrap resampling can be performed for this metric.
    """
    bootstrappable = True

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
        return permutation_test(x, y, lambda x, y: self.distance(x, y, **kwargs))

    def interval(self, x: pd.Series, y: pd.Series, level: float = 0.95, **kwargs) -> ConfidenceInterval:
        """
        Return a frequentist confidence interval for this metric obtained, via bootstrap resampling.

        Args:
            level: The confidence level of the interval, i.e the fraction of intervals
                   that will contain the true distance estimate.

            **kwargs: Optional keyword arguments to synthesized.insight.metrics.distance.bootstrap_interval.
        """
        if not self.bootstrappable:
            raise TypeError("Unable to perform bootstrap resampling of this metric.")
        else:
            interval = bootstrap_interval((x, y), lambda x, y: self.distance(x, y, **kwargs))
            return ConfidenceInterval(interval, level)


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
    bootstrappable = False

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
