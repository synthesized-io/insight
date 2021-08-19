"""
Collection of Metrics that measure the distance, or similarity, between two datasets.
"""
import math
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, kruskal, ks_2samp, wasserstein_distance

from ..check import ColumnCheck
from .base import BinnedMetricStatistics, MetricStatistics, OneColumnMetric
from .utils import (ConfidenceInterval, MetricStatisticsResult, affine_mean,
                    affine_stddev, binominal_proportion_interval,
                    binominal_proportion_p_value, bootstrap_pvalue,
                    bootstrap_statistic, infer_distr_type, standard_error,
                    zipped_hist)


class Mean(OneColumnMetric):
    name = "mean"

    def __init__(self,
                 check: ColumnCheck = None,
                 compute_p_val: bool = True,
                 compute_interval: bool = True,
                 confidence_level: float = 0.95):
        super().__init__(check)
        self.compute_p_val: bool = compute_p_val
        self.compute_interval: bool = compute_interval
        self.confidence_level: float = confidence_level

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr: pd.Series):
        if not check.affine(sr):
            return False
        return True

    def _compute_p_value(self, sr: pd.Series, metric_value: float):
        """
        Return a two-sided p-value for this metric using a bootstrapped distribution
        of the null hypothesis.

        Returns:
            The p-value under the null hypothesis.
        """
        ts_distribution = bootstrap_statistic((sr,), self._mean_call, n_samples=1000)
        return bootstrap_pvalue(metric_value, ts_distribution)

    def _compute_interval(self, sr: pd.Series, metric_value: float) -> ConfidenceInterval:
        """
        Return a confidence interval for this metric.

        Args:
            sr: Sample value
            metric_value: Value of the metric computed on the sample

        Returns:
            The confidence interval.
        """
        st_error = standard_error(sr)
        z_score = st.norm.ppf(self.confidence_level)
        return ConfidenceInterval((metric_value - z_score * st_error,
                                   metric_value + z_score * st_error), self.confidence_level)

    def _compute_metric(self, sr: pd.Series):
        """
        Compute the metric, p-value and the confidence interval

        Args:
            sr: Sample values

        Returns:
            MetricStatisticsResult object contain metric value, p-value and the confidence interval
        """
        mean = self._mean_call(sr)
        result = MetricStatisticsResult(metric_value=mean)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(sr, mean)
        if self.compute_interval:
            result.interval = self._compute_interval(sr, mean)
        return result

    def _mean_call(self, sr) -> float:
        return affine_mean(sr)


class StandardDeviation(OneColumnMetric):
    name = "standard_deviation"

    def __init__(self,
                 check: ColumnCheck = None,
                 compute_p_val: bool = True,
                 compute_interval: bool = True,
                 confidence_level: float = 0.95,
                 remove_outliers: float = 0.0):
        super().__init__(check)
        self.compute_p_val: bool = compute_p_val
        self.compute_interval: bool = compute_interval
        self.confidence_level: float = confidence_level
        self.remove_outliers: float = remove_outliers

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr: pd.Series):
        if not check.affine(sr):
            return False
        return True

    def _compute_p_value(self, sr: pd.Series, metric_value: float):
        """
        Return a two-sided p-value for this metric using a bootstrapped distribution
        of the null hypothesis.

        Returns:
            The p-value under the null hypothesis.
        """
        ts_distribution = bootstrap_statistic((sr,), self._stdv_call, n_samples=1000)
        return bootstrap_pvalue(metric_value, ts_distribution)

    def _compute_interval(self, sr: pd.Series, metric_value: float) -> ConfidenceInterval:
        """
        Return a frequentist confidence interval for this metric obtained, via bootstrap resampling.

        Args:
            sr: Sample values
            metric_value: Value of the metric computed on the sample

        Returns:
            The confidence interval.
        """
        n = len(sr)
        st_error = standard_error(sr)
        alpha = 1 - self.confidence_level
        left_critical_val = st.chi2.ppf(alpha / 2, n - 1)
        right_critical_val = st.chi2.ppf(1 - (alpha / 2), n - 1)
        return ConfidenceInterval((st_error * math.sqrt((n - 1) / left_critical_val),
                                   st_error * math.sqrt((n - 1) / right_critical_val)), self.confidence_level)

    def _compute_metric(self, sr: pd.Series):
        """
        Compute the metric, p-value and the confidence interval

        Args:
            sr: Sample values

        Returns:
            MetricStatisticsResult object containing metric value, p-value and the confidence interval
        """
        stdv = self._stdv_call(sr)
        result = MetricStatisticsResult(metric_value=stdv)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(sr, stdv)
        if self.compute_interval:
            result.interval = self._compute_interval(sr, stdv)
        return result

    def _stdv_call(self, sr: pd.Series):
        values = np.sort(sr.values)
        values = values[int(len(sr) * self.remove_outliers):int(len(sr) * (1.0 - self.remove_outliers))]

        return affine_stddev(pd.Series(values, name=sr.name))


class BinomialDistance(MetricStatistics):
    """Binomial distance statistics between two binary variables.

    The metric value ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "binomial_distance"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        return infer_distr_type(pd.concat((sr_a, sr_b))).is_binary()

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a numerical variable.
            sr_b (pd.Series): values of numerical variable.

        Returns:
            MetricStatisticsResult object containing the binomial distance between sr_a and sr_b,
            p-value and the confidence interval.
        """
        metric_value = sr_a.mean() - sr_b.mean()
        result = MetricStatisticsResult(metric_value)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(sr_a, sr_b, metric_value)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, metric_value)
        return result

    def _compute_p_value(self,
                         sr_a: pd.Series,
                         sr_b: pd.Series,
                         metric_value: float) -> float:
        """Calculate a p-value for the null hypothesis that the probability of success is p_y"""
        p_obs = sr_a.mean()
        p_null = sr_b.mean()
        n = len(sr_a)
        return binominal_proportion_p_value(p_obs, p_null, n)

    def _compute_interval(self,
                          sr_a: pd.Series,
                          sr_b: pd.Series,
                          metric_value: float,
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
        interval = binominal_proportion_interval(p, n, self.confidence_level, method)
        interval.value = interval.value[0] - sr_b.mean(), interval.value[1] - sr_b.mean()
        return interval


class KolmogorovSmirnovDistance(MetricStatistics):
    """Kolmogorov-Smirnov statistic between two continuous variables.

    The metric value ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "kolmogorov_smirnov_distance"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        if not check.continuous(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a continuous variable.
            sr_b (pd.Series): values of another continuous variable to compare.

        Returns:
            MetricStatisticsResult object containing the Kolmogorov-Smirnov distance between sr_a and sr_b,
            p-value and the confidence interval.
        """
        column_old_clean = pd.to_numeric(sr_a, errors='coerce').dropna()
        column_new_clean = pd.to_numeric(sr_b, errors='coerce').dropna()
        if len(column_old_clean) == 0 or len(column_new_clean) == 0:
            return MetricStatisticsResult(np.nan, None, None)

        distance, p_value = ks_2samp(column_old_clean, column_new_clean)
        result = MetricStatisticsResult(metric_value=distance, p_value=p_value)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, distance)
        return result


class KruskalWallis(MetricStatistics):
    """Kruskal Wallis distance statistics between two numerical variables.

    The metric value ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "kruskal_wallis"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.continuous(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_interval(self,
                          sr_a: pd.Series,
                          sr_b: pd.Series,
                          metric_value: float):
        raise NotImplementedError

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a numerical variable.
            sr_b (pd.Series): values of numerical variable.

        Returns:
            MetricStatisticsResult object containing the Kruskal-Wallis distance between sr_a and sr_b,
            p-value and the confidence interval.
        """
        distance, p_value = kruskal(sr_a, sr_b)
        result = MetricStatisticsResult(metric_value=distance, p_value=p_value)
        return result


class EarthMoversDistance(MetricStatistics):
    """Earth mover's distance (aka 1-Wasserstein distance) statistics between two nominal variables.

    The metric value ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "earth_movers_distance"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        if not check.categorical(sr_a) or not check.categorical(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a nominal variable.
            sr_b (pd.Series): values of another nominal variable to compare.

        Returns:
            MetricStatisticsResult object containing the earth mover's distance between sr_a and sr_b,
            p-value and the confidence interval.
        """
        old = sr_a.to_numpy().astype(str)
        new = sr_b.to_numpy().astype(str)

        space = set(old).union(set(new))
        if len(space) > 1e4:
            return MetricStatisticsResult(np.nan, None, None)

        old_unique, counts = np.unique(old, return_counts=True)
        old_counts = dict(zip(old_unique, counts))

        new_unique, counts = np.unique(new, return_counts=True)
        new_counts = dict(zip(new_unique, counts))

        p = np.array([float(old_counts[x]) if x in old_counts else 0.0 for x in space])
        q = np.array([float(new_counts[x]) if x in new_counts else 0.0 for x in space])

        p /= np.sum(p)
        q /= np.sum(q)

        distance = 0.5 * np.sum(np.abs(p.astype(np.float64) - q.astype(np.float64)))

        result = MetricStatisticsResult(metric_value=distance)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(sr_a, sr_b, distance)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, distance)
        return result


class EarthMoversDistanceBinned(BinnedMetricStatistics):
    """Earth mover's distance (aka 1-Wasserstein distance) statistics between two nominal variables.

    The histograms can represent counts of nominal categories or counts on
    an ordinal range. If the latter, they must have equal binning.

    The metric value ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "earth_movers_distance_binned"

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a nominal variable.
            sr_b (pd.Series): values of another nominal variable to compare.

        Returns:
            MetricStatisticsResult object containing the earth mover's binned distance between sr_a and sr_b,
            p-value and the confidence interval.
        """
        if sr_a.sum() == 0 and sr_b.sum() == 0:
            return MetricStatisticsResult(0., None, None)
        elif sr_a.sum() == 0 or sr_b.sum() == 0:
            return MetricStatisticsResult(1., None, None)

        # normalise counts for consistency with scipy.stats.wasserstein
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(sr_a / sr_a.sum())
            y = np.nan_to_num(sr_b / sr_b.sum())

        if self.bins is None:
            # if bins not given, histograms are assumed to be counts of nominal categories,
            # and therefore distances betwen bins are meaningless. Set to all distances to
            # unity to model this.
            distance = 0.5 * np.sum(np.abs(x.astype(np.float64) - y.astype(np.float64)))
        else:
            # otherwise, use pair-wise euclidean distances between bin centers for scale data
            bin_centers = self.bins[:-1] + np.diff(self.bins) / 2.
            distance = wasserstein_distance(bin_centers, bin_centers, u_weights=x, v_weights=y)

        result = MetricStatisticsResult(metric_value=distance)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(sr_a, sr_b, distance)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, distance)
        return result


class HellingerDistance(BinnedMetricStatistics):
    """Hellinger distance statistics between samples from two distributions.

    Samples are binned during the computation to approximate the pdfs P(x) and P(y).

    The metric value ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "hellinger_distance"

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a nominal variable.
            sr_b (pd.Series): values of another nominal variable to compare.

        Returns:
            MetricStatisticsResult object containing the hellinger distance between sr_a and sr_b,
            p-value and the confidence interval.
        """
        bin_edges = [bin[0] for bin in self.bins] if self.bins else None
        (x, y), _ = zipped_hist((sr_a, sr_b), bin_edges=bin_edges, ret_bins=True)
        distance = np.linalg.norm(np.sqrt(x) - np.sqrt(y)) / np.sqrt(2)

        result = MetricStatisticsResult(metric_value=distance)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(x, y, distance)
        if self.compute_interval:
            result.interval = self._compute_interval(x, y, distance)
        return result


class KullbackLeiblerDivergence(BinnedMetricStatistics):
    """Kullback–Leibler Divergence or Relative Entropy statistics between two probability distributions.

    The metric value ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "kullback_leibler_divergence"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        x_dtype = str(check.infer_dtype(sr_a).dtype)
        y_dtype = str(check.infer_dtype(sr_b).dtype)

        return x_dtype == y_dtype

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            MetricStatisticsResult object containing the kullback-leibler divergence between sr_a and sr_b,
            p-value and the confidence interval.
        """
        bin_edges = [bin[0] for bin in self.bins] if self.bins else None
        (x, y), _ = zipped_hist((sr_a, sr_b), bin_edges=bin_edges, ret_bins=True)
        divergence = entropy(np.array(x), np.array(y))

        result = MetricStatisticsResult(metric_value=divergence)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(x, y, divergence)
        if self.compute_interval:
            result.interval = self._compute_interval(x, y, divergence)
        return result


class JensenShannonDivergence(BinnedMetricStatistics):
    """Jensen-Shannon Divergence statistics between two probability distributions.

    The metric value ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "jensen_shannon_divergence"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        x_dtype = str(check.infer_dtype(sr_a).dtype)
        y_dtype = str(check.infer_dtype(sr_b).dtype)

        return x_dtype == y_dtype

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            MetricStatisticsResult object containing the jensen-shannon divergence between sr_a and sr_b,
            p-value and the confidence interval.
        """
        bin_edges = [bin[0] for bin in self.bins] if self.bins else None
        (x, y), _ = zipped_hist((sr_a, sr_b), bin_edges=bin_edges, ret_bins=True)
        divergence = jensenshannon(x, y)

        result = MetricStatisticsResult(metric_value=divergence)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(x, y, divergence)
        if self.compute_interval:
            result.interval = self._compute_interval(x, y, divergence)
        return result


class Norm(BinnedMetricStatistics):
    """Norm statistics between two probability distributions.

    The metric value ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.

    Args:
        ord (Union[str, int], optional):
                The order of the norm. Possible values include positive numbers, 'fro', 'nuc'.
                See numpy.linalg.norm for more details. Defaults to 2.
    """
    name = "norm"

    def __init__(self,
                 check: ColumnCheck = None,
                 compute_p_val: bool = True,
                 compute_interval: bool = True,
                 confidence_level: float = 0.95,
                 bootstrap_mode: bool = False,
                 bins: Optional[Sequence[Union[float, int]]] = None,
                 ord: Union[str, int] = 2):
        super().__init__(check, compute_p_val, compute_interval, confidence_level, bootstrap_mode, bins)
        self.ord = ord

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        x_dtype = str(check.infer_dtype(sr_a).dtype)
        y_dtype = str(check.infer_dtype(sr_b).dtype)

        return x_dtype == y_dtype

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            MetricStatisticsResult object containing the lp-norm between sr_a and sr_b,
            p-value and the confidence interval.
        """
        bin_edges = [bin[0] for bin in self.bins] if self.bins else None
        (x, y), _ = zipped_hist((sr_a, sr_b), bin_edges=bin_edges, ret_bins=True)
        norm_val = np.linalg.norm(x - y, ord=self.ord)

        result = MetricStatisticsResult(metric_value=norm_val)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(x, y, norm_val)
        if self.compute_interval:
            result.interval = self._compute_interval(x, y, norm_val)
        return result
