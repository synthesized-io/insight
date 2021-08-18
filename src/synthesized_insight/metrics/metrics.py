"""
Collection of Metrics that measure the distance, or similarity, between two datasets.
"""
import math
import warnings
from typing import Any, Optional, Sequence, Union

import dcor as dcor
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
from scipy.stats import (entropy, kendalltau, kruskal, ks_2samp, spearmanr,
                         wasserstein_distance)
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ..check import ColumnCheck
from .base import OneColumnMetric, TwoColumnMetric
from .utils import (ConfidenceInterval, MetricStatisticsResult, affine_mean,
                    affine_stddev, binominal_proportion_interval,
                    binominal_proportion_p_value, bootstrap_binned_statistic,
                    bootstrap_interval, bootstrap_pvalue, bootstrap_statistic,
                    infer_distr_type, permutation_test, standard_error,
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


class KendallTauCorrelation(MetricStatistics):
    """Kendall's Tau correlation coefficient statistics between ordinal variables.

    The cofficient value ranges from -1 to 1, indicating the strength and direction of the relationship
    between the two variables.

    Args:
        max_p_value (float, optional): Returns None if p-value from test is above this threshold.
    """
    name = "kendall_tau_correlation"
    symmetric = True

    def __init__(self,
                 check: ColumnCheck = None,
                 compute_p_val: bool = True,
                 compute_interval: bool = True,
                 confidence_level: float = 0.95,
                 bootstrap_mode: bool = False,
                 max_p_value: float = 1.0
                 ):
        super().__init__(check, compute_p_val, compute_interval, confidence_level, bootstrap_mode)
        self.max_p_value = max_p_value

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        # Given columns should be both categorical or both ordinal
        if ((check.ordinal(sr_a) and check.ordinal(sr_b))
           or (check.categorical(sr_a) and check.categorical(sr_b))):
            return True
        return False

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            MetricStatisticsResult object containing the coefficient between sr_a and sr_b,
            p-value and the confidence interval.
        """
        sr_a = pd.to_numeric(sr_a, errors='coerce')
        sr_b = pd.to_numeric(sr_b, errors='coerce')

        corr, p_value = kendalltau(sr_a.values, sr_b.values, nan_policy='omit')
        result = MetricStatisticsResult(metric_value=corr if p_value <= self.max_p_value else None,
                                        p_value=p_value)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, corr)
        return result


class SpearmanRhoCorrelation(MetricStatistics):
    """Spearman's rank correlation coefficient statistics between ordinal variables.

    The cofficient value ranges from -1 to 1, measures the strength and direction of monotonic
    relationship between two ranked (ordinal) variables.

    Args:
        max_p_value (float, optional): Returns None if p-value from test is above this threshold.
    """
    name = "spearman_rho_correlation"

    def __init__(self,
                 check: ColumnCheck = None,
                 compute_p_val: bool = True,
                 compute_interval: bool = True,
                 confidence_level: float = 0.95,
                 bootstrap_mode: bool = False,
                 max_p_value: float = 1.0):
        super().__init__(check, compute_p_val, compute_interval, confidence_level, bootstrap_mode)
        self.max_p_value = max_p_value

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.ordinal(sr_a) or not check.ordinal(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            MetricStatisticsResult object containing the spearman coefficient between sr_a and sr_b,
            p-value and the confidence interval.
        """
        x = sr_a.values
        y = sr_b.values

        if self.check.infer_dtype(sr_a).dtype.kind == 'M':
            x = pd.to_numeric(pd.to_datetime(x, errors='coerce'), errors='coerce')
        if self.check.infer_dtype(sr_b).dtype.kind == 'M':
            y = pd.to_numeric(pd.to_datetime(y, errors='coerce'), errors='coerce')

        corr, p_value = spearmanr(x, y, nan_policy='omit')
        result = MetricStatisticsResult(metric_value=corr if p_value <= self.max_p_value else None,
                                        p_value=p_value)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, corr)
        return result


class CramersV(MetricStatistics):
    """Cramér's V correlation coefficient statistics between nominal variables.

    The metric value ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "cramers_v"
    symmetric = True

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.categorical(sr_a) or not check.categorical(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            MetricStatisticsResult object containing the CramersV coefficient between sr_a and sr_b,
            p-value and the confidence interval.
        """
        table_orig = pd.crosstab(sr_a.astype(str), sr_b.astype(str))
        table = np.asarray(table_orig, dtype=np.float64)

        if table.min() == 0:
            table[table == 0] = 0.5

        n = table.sum()
        row = table.sum(1) / n
        col = table.sum(0) / n

        row = pd.Series(data=row, index=table_orig.index)
        col = pd.Series(data=col, index=table_orig.columns)
        itab = np.outer(row, col)
        probs = pd.DataFrame(
            data=itab, index=table_orig.index, columns=table_orig.columns
        )

        fit = table.sum() * probs
        expected = fit.to_numpy()

        real = table
        r, c = real.shape
        n = np.sum(real)
        v = np.sum((real - expected) ** 2 / (expected * n * min(r - 1, c - 1))) ** 0.5

        result = MetricStatisticsResult(metric_value=v)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(sr_a, sr_b, v)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, v)
        return result


class R2Mcfadden(MetricStatistics):
    """R2 Mcfadden correlation coefficient statistics between categorical and numerical variables.

    It trains two multinomial logistic regression models on the data, one using the numerical
    series as the feature and the other only using the intercept term as the input.
    The categorical column is used for the target labels. It then calculates the null
    and the model likelihoods based on them, which are used to compute the pseudo-R2 McFadden score,
    which is used as a correlation coefficient.

    The coefficient value ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "r2_mcfadden"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.categorical(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_p_value(self,
                         sr_a: pd.Series,
                         sr_b: pd.Series,
                         metric_value: float):
        raise NotImplementedError

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a nominal variable.
            sr_b (pd.Series): values of numerical variable.

        Returns:
            MetricStatisticsResult object containing the R2 Mcfadden correlation coefficient between sr_a and sr_b,
            p-value and the confidence interval.
        """
        x = sr_b.to_numpy().reshape(-1, 1)
        x = StandardScaler().fit_transform(x)
        y = sr_a.to_numpy()

        enc = LabelEncoder()
        y = enc.fit_transform(y)

        lr_feature = linear_model.LogisticRegression()
        lr_feature.fit(x, y)

        y_one_hot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

        log_pred = lr_feature.predict_log_proba(x)
        ll_feature = np.sum(y_one_hot * log_pred)

        lr_intercept = linear_model.LogisticRegression()
        lr_intercept.fit(np.ones_like(y).reshape(-1, 1), y)

        log_pred = lr_intercept.predict_log_proba(x)
        ll_intercept = np.sum(y_one_hot * log_pred)

        pseudo_r2 = 1 - ll_feature / ll_intercept

        result = MetricStatisticsResult(metric_value=pseudo_r2)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, pseudo_r2)
        return result


class DistanceNNCorrelation(MetricStatistics):
    """Distance nn correlation coefficient statistics between two numerical variables.

    It uses non-linear correlation distance to obtain a correlation coefficient for
    numerical-numerical column pairs.

    The coefficient ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "distance_nn_correlation"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.continuous(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a numerical variable.
            sr_b (pd.Series): values of numerical variable.

        Returns:
            MetricStatisticsResult object containing the distance nn correlation coefficient between sr_a and sr_b,
            p-value and the confidence interval.
        """
        warnings.filterwarnings(action="ignore", category=UserWarning)

        if sr_a.size < sr_b.size:
            sr_a = sr_a.append(pd.Series(sr_a.mean()).repeat(sr_b.size - sr_a.size), ignore_index=True)
        elif sr_a.size > sr_b.size:
            sr_b = sr_b.append(pd.Series(sr_b.mean()).repeat(sr_a.size - sr_b.size), ignore_index=True)

        dcorr = dcor.distance_correlation(sr_a, sr_b)

        result = MetricStatisticsResult(metric_value=dcorr)
        if self.compute_p_val:
            result.p_value = self._compute_p_value(sr_a, sr_b, dcorr)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, dcorr)
        return result


class DistanceCNCorrelation(MetricStatistics):
    """Distance cn correlation coefficient statistics between categorical and numerical variables.

    It uses non-linear correlation distance to obtain a correlation coefficient for
    categorical-numerical column pairs.

    The coefficient value ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "distance_cn_correlation"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.categorical(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_p_value(self,
                         sr_a: pd.Series,
                         sr_b: pd.Series,
                         metric_value: float):
        raise NotImplementedError

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> MetricStatisticsResult:
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a categorical variable.
            sr_b (pd.Series): values of numerical variable.

        Returns:
            MetricStatisticsResult object containing the distance cn correlation coefficient between sr_a and sr_b,
            p-value and the confidence interval.
        """
        warnings.filterwarnings(action="ignore", category=UserWarning)
        sr_a_codes = sr_a.astype("category").cat.codes
        groups_obj = sr_b.groupby(sr_a_codes)
        arrays = [groups_obj.get_group(cat) for cat in sr_a_codes.unique() if cat in groups_obj.groups.keys()]

        total = 0.0
        n = len(arrays)

        for i in range(0, n):
            for j in range(i + 1, n):
                sr_i = arrays[i]
                sr_j = arrays[j]

                # Handle groups with a different number of elements.
                if sr_i.size < sr_j.size:
                    sr_i = sr_i.append(sr_i.sample(sr_j.size - sr_i.size, replace=True), ignore_index=True)
                elif sr_i.size > sr_j.size:
                    sr_j = sr_j.append(sr_j.sample(sr_i.size - sr_j.size, replace=True), ignore_index=True)
                total += dcor.distance_correlation(sr_i, sr_j)

        if n > 1:
            total /= n * (n - 1) / 2

        result = MetricStatisticsResult(metric_value=total)
        if self.compute_interval:
            result.interval = self._compute_interval(sr_a, sr_b, total)
        return result
