from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import binom_test, kendalltau, kruskal, ks_2samp, spearmanr

from ..check import Check, ColumnCheck
from .base import TwoColumnMetric, TwoColumnTest
from .utils import bootstrap_binned_statistic, bootstrap_statistic


class BinomialDistanceTest(TwoColumnTest):
    """Binomial distance test between two binary variables.

    The metric value ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "binomial_distance"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        return sr_a.append(sr_b).nunique() == 2

    def _compute_test(self, sr_a: pd.Series, sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """Calculate binomial metric and exact p-value for an observed binomial proportion of a sample."""

        metric_value = sr_a.mean() - sr_b.mean()

        p_obs = sr_a.mean()
        p_null = sr_b.mean()
        n = len(sr_a)

        k = np.ceil(p_obs * n)
        p_value = binom_test(k, n, p_null, self.alternative)
        return metric_value, p_value


class KolmogorovSmirnovDistanceTest(TwoColumnTest):
    """Kolmogorov-Smirnov distance test between two continuous variables.
    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.

    """
    name = "kolmogorov_smirnov_distance"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        if not check.continuous(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_test(self, sr_a: pd.Series, sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """Calculate the metric and p-value.
        Args:
            sr_a (pd.Series): values of a continuous variable.
            sr_b (pd.Series): values of another continuous variable to compare.
        Returns:
            The Kolmogorov-Smirnov distance between sr_a and sr_b,
            and p-value on the test statistics
        """
        column_old_clean = pd.to_numeric(sr_a, errors='coerce').dropna()
        column_new_clean = pd.to_numeric(sr_b, errors='coerce').dropna()
        if len(column_old_clean) == 0 or len(column_new_clean) == 0:
            return np.nan, np.nan
        ks_distance, p_value = ks_2samp(column_old_clean, column_new_clean, alternative=self.alternative)
        return ks_distance, p_value


class KruskalWallisTest(TwoColumnTest):
    """Kruskal Wallis distance test between two numerical variables.

    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "kruskal_wallis"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        if not check.continuous(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_test(self, sr_a: pd.Series, sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """Calculate the metric and the p_value.
        Args:
            sr_a (pd.Series): values of a numerical variable.
            sr_b (pd.Series): values of numerical variable.
        Returns:
            The Kruskal Wallis distance between sr_a and sr_b,
            and p-value on the test statistics
        """
        metric_value, p_value = kruskal(sr_a, sr_b)
        return metric_value, p_value


class KendallTauCorrelationTest(TwoColumnTest):
    """Kendall's Tau correlation coefficient test between ordinal variables.
    The statistic ranges from -1 to 1, indicating the strength and direction of the relationship
    between the two variables.

    Args:
        max_p_value (float, optional): Returns None if p-value from test is above this threshold.
    """
    name = "kendall_tau_correlation"
    symmetric = True

    def __init__(self,
                 check: Check = ColumnCheck(),
                 alternative: str = 'two-sided',
                 max_p_value: float = 1.0):
        super().__init__(check, alternative)
        self.max_p_value = max_p_value

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        # Given columns should be both categorical or both ordinal
        if (check.ordinal(sr_a) and check.ordinal(sr_b)):
            return True
        return False

    def _compute_test(self, sr_a: pd.Series, sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """Returns kendall-tau correlation and p-value"""

        sr_a = pd.to_numeric(sr_a, errors='coerce')
        sr_b = pd.to_numeric(sr_b, errors='coerce')
        corr, p_value = kendalltau(sr_a.values, sr_b.values, nan_policy='omit')
        if p_value is not None and p_value <= self.max_p_value:
            return corr, p_value
        else:
            return None, p_value


class SpearmanRhoCorrelationTest(TwoColumnTest):
    """Spearman's rank correlation coefficient test between ordinal variables.
    The statistic ranges from -1 to 1, measures the strength and direction of monotonic
    relationship between two ranked (ordinal) variables.

    Args:
        max_p_value (float, optional): Returns None if p-value from test is above this threshold.
    """
    name = "spearman_rho_correlation"

    def __init__(self,
                 check: Check = ColumnCheck(),
                 alternative: str = 'two-sided',
                 max_p_value: float = 1.0):
        super().__init__(check, alternative)
        self.max_p_value = max_p_value

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        if not check.ordinal(sr_a) or not check.ordinal(sr_b):
            return False
        return True

    def _compute_test(self, sr_a: pd.Series, sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """Return spearman-rho correlation and p-value"""

        x = sr_a.values
        y = sr_b.values
        if self.check.infer_dtype(sr_a).dtype.kind == 'M':
            x = pd.to_numeric(pd.to_datetime(x, errors='coerce'), errors='coerce')
        if self.check.infer_dtype(sr_b).dtype.kind == 'M':
            y = pd.to_numeric(pd.to_datetime(y, errors='coerce'), errors='coerce')
        corr, p_value = spearmanr(x, y, nan_policy='omit')
        if p_value is not None and p_value <= self.max_p_value:
            return corr, p_value
        else:
            return None, p_value


class BootstrapTest:
    """Performs bootstrap test

    Args:
        metric_cls_obj: The instantiated metric object for which the test if performed
        binned: If binning is done
    """
    name = "bootstrap_test"

    def __init__(self,
                 metric_cls_obj: TwoColumnMetric,
                 alternative: str = 'two-sided',
                 binned: bool = False):
        self.alternative = alternative
        self.metric_cls_obj = metric_cls_obj
        self.binned = binned

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        return True  # metric object provided during initialization will perform the check while computing the metrics

    def _bootstrap_pvalue(self,
                          t_obs: float,
                          t_distribution: np.ndarray) -> float:
        """
        Calculate a p-value using a bootstrapped test statistic distribution

        Args:
            t_obs: Observed value of the test statistic.
            t_distribution: Samples of test statistic distribution under the null hypothesis.

        Returns:
            The p-value under the null hypothesis.
        """

        n_samples = len(t_distribution)

        if self.alternative == 'two-sided':
            p = np.sum(np.abs(t_distribution) >= np.abs(t_obs)) / n_samples

        elif self.alternative == 'greater':
            p = np.sum(t_distribution >= t_obs) / n_samples

        else:
            p = np.sum(t_distribution < t_obs) / n_samples

        return p

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """Return metric_value and p-value for this metric using a bootstrap test.

        The null hypothesis is that both data samples are from the same distribution."""

        metric_value = self.metric_cls_obj(sr_a, sr_b)
        if metric_value is None:
            return None, None

        if not self.binned:
            ts_distribution = bootstrap_statistic((sr_a, sr_b),
                                                  lambda x, y: self.metric_cls_obj(x, y),
                                                  n_samples=1000)
        else:
            ts_distribution = bootstrap_binned_statistic((sr_a, sr_b),
                                                         lambda x, y: self.metric_cls_obj(x, y),
                                                         n_samples=1000)
        p_value = self._bootstrap_pvalue(metric_value, ts_distribution)
        return metric_value, p_value


class PermutationTest:
    """Performs permutation test

    Args:
        metric_cls_obj: The instantiated metric object for which the test if performed
        n_perms: No. of permutations to be done
    """
    name = "permutation_test"

    def __init__(self,
                 metric_cls_obj: TwoColumnMetric,
                 alternative: str = 'two-sided',
                 n_perms: int = 100):
        self.alternative = alternative
        self.metric_cls_obj = metric_cls_obj
        self.n_perms = n_perms

    def _permutation_test(self,
                          x: pd.Series,
                          y: pd.Series,
                          t: Callable[[pd.Series, pd.Series], float]) -> float:
        """
        Perform a two sample permutation test.
        Determines the probability of observing t(x, y) or greater under the null hypothesis that x
        and y are from the same distribution.
        Args:
            x: First data sample.
            y: Second data sample.
            t: Callable that returns the test statistic.
        Returns:
            The p-value of t_obs under the null hypothesis.
        """

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

        if self.alternative == 'two-sided':
            p = np.sum(np.abs(t_null) >= np.abs(t_obs)) / self.n_perms

        elif self.alternative == 'greater':
            p = np.sum(t_null >= t_obs) / self.n_perms

        else:
            p = np.sum(t_null < t_obs) / self.n_perms

        return p

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        """Return metric_value and p-value for this metric using a permutation test.

        The null hypothesis is that both data samples are from the same distribution."""

        metric_value = self.metric_cls_obj(sr_a, sr_b)
        if metric_value is None:
            return None, None

        p_value = self._permutation_test(sr_a,
                                         sr_b,
                                         lambda x, y: self.metric_cls_obj(x, y))
        return metric_value, p_value
