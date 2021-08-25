from typing import Callable, Union

import numpy as np
import pandas as pd
from scipy.stats import binom_test, kendalltau, kruskal, ks_2samp, spearmanr

from ..check import ColumnCheck
from .base import TwoColumnMetric, TwoColumnMetricTest
from .utils import bootstrap_binned_statistic, bootstrap_statistic, infer_distr_type


class BinomialDistanceTest(TwoColumnMetricTest):
    """Binomial distance test between two binary variables.
    The metric value ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "binomial_distance"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        return infer_distr_type(pd.concat((sr_a, sr_b))).is_binary()

    def _binominal_proportion_p_value(self,
                                      p_obs: float,
                                      p_null: float,
                                      n: int,
                                      alternative: str = 'two-sided'):
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
        self.p_value = binom_test(k, n, p_null, alternative)
        return self.p_value

    def _compute_p_value(self,
                         sr_a: pd.Series,
                         sr_b: pd.Series,
                         metric_value: Union[int, float, None],
                         alternative: str = 'two-sided') -> Union[int, float, None]:
        """Calculate a p-value for the null hypothesis that the probability of success is p_y"""
        p_obs = sr_a.mean()
        p_null = sr_b.mean()
        n = len(sr_a)
        return self._binominal_proportion_p_value(p_obs, p_null, n, alternative)

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        return sr_a.mean() - sr_b.mean()


class KolmogorovSmirnovDistanceTest(TwoColumnMetricTest):
    """Kolmogorov-Smirnov distance test between two continuous variables.
    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.

    """
    name = "kolmogorov_smirnov_distance"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        if not check.continuous(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.
        Args:
            sr_a (pd.Series): values of a continuous variable.
            sr_b (pd.Series): values of another continuous variable to compare.
        Returns:
            The Kolmogorov-Smirnov distance between sr_a and sr_b, and p-value
        """
        column_old_clean = pd.to_numeric(sr_a, errors='coerce').dropna()
        column_new_clean = pd.to_numeric(sr_b, errors='coerce').dropna()
        if len(column_old_clean) == 0 or len(column_new_clean) == 0:
            return np.nan
        ks_distance, self.p_value = ks_2samp(column_old_clean, column_new_clean)
        return ks_distance


class KruskalWallisTest(TwoColumnMetricTest):
    """Kruskal Wallis distance test between two numerical variables.
    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "kruskal_wallis"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.continuous(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.
        Args:
            sr_a (pd.Series): values of a numerical variable.
            sr_b (pd.Series): values of numerical variable.
        Returns:
            The Kruskal Wallis distance between sr_a and sr_b.
        """
        metric_value, self.p_value = kruskal(sr_a, sr_b)
        return metric_value


class KendallTauCorrelationTest(TwoColumnMetricTest):
    """Kendall's Tau correlation coefficient test between ordinal variables.
    The statistic ranges from -1 to 1, indicating the strength and direction of the relationship
    between the two variables.

    Args:
        max_p_value (float, optional): Returns None if p-value from test is above this threshold.
    """
    name = "kendall_tau_correlation"
    symmetric = True

    def __init__(self,
                 check: ColumnCheck = None,
                 metric_cls_obj: TwoColumnMetric = None,
                 max_p_value: float = 1.0):
        super().__init__(check, metric_cls_obj)
        self.max_p_value = max_p_value

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        # Given columns should be both categorical or both ordinal
        if (check.ordinal(sr_a) and not check.ordinal(sr_b))\
            or (not check.ordinal(sr_a) and check.ordinal(sr_b))\
            or (check.categorical(sr_a) and not check.categorical(sr_b))\
                or (not check.categorical(sr_a) and check.categorical(sr_b)):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        sr_a = pd.to_numeric(sr_a, errors='coerce')
        sr_b = pd.to_numeric(sr_b, errors='coerce')
        corr, self.p_value = kendalltau(sr_a.values, sr_b.values, nan_policy='omit')
        print(self.max_p_value, self.p_value, corr)
        if self.p_value is not None and self.p_value <= self.max_p_value:
            return corr
        else:
            return None


class SpearmanRhoCorrelationTest(TwoColumnMetricTest):
    """Spearman's rank correlation coefficient test between ordinal variables.
    The statistic ranges from -1 to 1, measures the strength and direction of monotonic
    relationship between two ranked (ordinal) variables.

    Args:
        max_p_value (float, optional): Returns None if p-value from test is above this threshold.
    """
    name = "spearman_rho_correlation"

    def __init__(self,
                 check: ColumnCheck = None,
                 metric_cls_obj: TwoColumnMetric = None,
                 max_p_value: float = 1.0):
        super().__init__(check, metric_cls_obj)
        self.max_p_value = max_p_value

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.ordinal(sr_a) or not check.ordinal(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        x = sr_a.values
        y = sr_b.values
        if self.check.infer_dtype(sr_a).dtype.kind == 'M':
            x = pd.to_numeric(pd.to_datetime(x, errors='coerce'), errors='coerce')
        if self.check.infer_dtype(sr_b).dtype.kind == 'M':
            y = pd.to_numeric(pd.to_datetime(y, errors='coerce'), errors='coerce')
        corr, self.p_value = spearmanr(x, y, nan_policy='omit')
        if self.p_value is not None and self.p_value <= self.max_p_value:
            return corr
        else:
            return None


class BootstrapTest(TwoColumnMetricTest):
    def __init__(self,
                 check: ColumnCheck = None,
                 metric_cls_obj: TwoColumnMetric = None,
                 binned: bool = False):
        if metric_cls_obj is None:
            raise ValueError("Metric class object can't be Null")
        super().__init__(check, metric_cls_obj)
        self.binned = binned

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        return True  # metric object provided during initialization will perform the check while computing the metrics

    def _bootstrap_pvalue(self,
                          t_obs: float,
                          t_distribution: pd.Series,
                          alternative: str = 'two-sided'):
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

    def _compute_p_value(self,
                         sr_a: pd.Series,
                         sr_b: pd.Series,
                         metric_value: float,
                         alternative: str = 'two-sided') -> Union[int, float, None]:
        # need to add this because of mypy bug/issue (#2608) which doesn't recognize that self.metric_cls_obj has already been checked against None
        # callable: Callable[[pd.Series, pd.Series], float] = self.metric_cls_obj

        if not self.binned:
            ts_distribution = bootstrap_statistic((sr_a, sr_b),
                                                  lambda x, y: self._compute_metric(x, y),
                                                  n_samples=1000)
        else:
            ts_distribution = bootstrap_binned_statistic((sr_a, sr_b),
                                                         lambda x, y: self._compute_metric(x, y),
                                                         n_samples=1000)
        self.p_value = self._bootstrap_pvalue(metric_value, ts_distribution, alternative)
        return self.p_value

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        return self.metric_cls_obj(sr_a, sr_b) if self.metric_cls_obj else None


class PermutationTest(TwoColumnMetricTest):
    def __init__(self,
                 check: ColumnCheck = None,
                 metric_cls_obj: TwoColumnMetric = None,
                 n_perms: int = 100):
        if metric_cls_obj is None:
            raise ValueError("Metric class object can't be Null")
        super().__init__(check, metric_cls_obj)
        self.n_perms = n_perms

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        return True  # metric object provided during initialization will perform the check while computing the metrics

    def _permutation_test(self,
                          x: pd.Series,
                          y: pd.Series,
                          t: Callable[[pd.Series, pd.Series], float],
                          alternative: str = 'two-sided'):
        """
        Perform a two sample permutation test.
        Determines the probability of observing t(x, y) or greater under the null hypothesis that x
        and y are from the same distribution.
        Args:
            x: First data sample.
            y: Second data sample.
            t: Callable that returns the test statistic.
            alternative: Optional; Indicates the alternative hypothesis.
                One of 'two-sided', 'greater' ,'less'
        Returns:
            The p-value of t_obs under the null hypothesis.
        """

        if alternative not in ('two-sided', 'greater', 'less'):
            raise ValueError("'alternative' argument must be one of 'two-sided', 'greater', 'less'")

        if t is None:
            raise ValueError("Callable function required, can't be None")

        t_obs = t(x, y)
        pooled_data = np.concatenate((x, y))
        t_null = np.empty(self.n_perms)

        for i in range(self.n_perms):
            perm = np.random.permutation(pooled_data)
            x_sample = perm[:len(x)]
            y_sample = perm[len(x):]
            t_null[i] = t(pd.Series(x_sample), pd.Series(y_sample))

        if alternative == 'two-sided':
            p = np.sum(np.abs(t_null) >= np.abs(t_obs)) / self.n_perms

        elif alternative == 'greater':
            p = np.sum(t_null >= t_obs) / self.n_perms

        else:
            p = np.sum(t_null < t_obs) / self.n_perms

        return p

    def _compute_p_value(self,
                         sr_a: pd.Series,
                         sr_b: pd.Series,
                         metric_value: Union[int, float, None],
                         alternative: str = 'two-sided') -> Union[int, float, None]:
        """Return a p-value for this metric using a permutation test. The null hypothesis
        is that both data samples are from the same distribution."""

        # # need to add this because of mypy bug/issue (#2608) which doesn't recognize that self.metric_cls_obj has already been checked against None
        # callable: Callable[[pd.Series, pd.Series], float] = self.metric_cls_obj
        self.p_value = self._permutation_test(sr_a,
                                              sr_b,
                                              lambda x, y: self._compute_metric(x, y),
                                              alternative)
        return self.p_value

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        return self.metric_cls_obj(sr_a, sr_b) if self.metric_cls_obj else None
