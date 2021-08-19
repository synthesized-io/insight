import numpy as np
import pandas as pd
from scipy.stats import kendalltau, ks_2samp, spearmanr

from ..check import ColumnCheck
from .base import OneColumnMetric, TwoColumnMetric


class Mean(OneColumnMetric):
    name = "mean"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr: pd.Series):
        if not check.affine(sr):
            return False
        return True

    def _compute_metric(self, sr: pd.Series):
        return affine_mean(sr)


class StandardDeviation(OneColumnMetric):
    name = "standard_deviation"

    def __init__(self, check: ColumnCheck = None, remove_outliers: float = 0.0):
        super().__init__(check)
        self.remove_outliers = remove_outliers

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr: pd.Series):
        if not check.affine(sr):
            return False
        return True

    def _compute_metric(self, sr: pd.Series):
        values = np.sort(sr.values)
        values = values[int(len(sr) * self.remove_outliers):int(len(sr) * (1.0 - self.remove_outliers))]

        return affine_stddev(pd.Series(values, name=sr.name))


class KendallTauCorrelation(TwoColumnMetric):
    """Kendall's Tau correlation coefficient between ordinal variables.

    The statistic ranges from -1 to 1, indicating the strength and direction of the relationship
    between the two variables.

    Args:
        max_p_value (float, optional): Returns None if p-value from test is above this threshold.
    """
    name = "kendall_tau_correlation"
    symmetric = True

    def __init__(self, check: ColumnCheck = None, max_p_value: float = 1.0, calculate_categorical: bool = False):
        super().__init__(check)
        self.max_p_value = max_p_value
        self.calculate_categorical = calculate_categorical

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

        corr, p_value = kendalltau(sr_a.values, sr_b.values, nan_policy='omit')

        if p_value <= self.max_p_value:
            return corr
        else:
            return None


class SpearmanRhoCorrelation(TwoColumnMetric):
    """Spearman's rank correlation coefficient between ordinal variables.

    The statistic ranges from -1 to 1, measures the strength and direction of monotonic
    relationship between two ranked (ordinal) variables.

    Args:
        max_p_value (float, optional): Returns None if p-value from test is above this threshold.
    """
    name = "spearman_rho_correlation"

    def __init__(self, check: ColumnCheck = None, max_p_value: float = 1.0):
        super().__init__(check)
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

        corr, p_value = spearmanr(x, y, nan_policy='omit')

        if p_value <= self.max_p_value:
            return corr
        else:
            return None


class CramersV(TwoColumnMetric):
    """Cramér's V correlation coefficient between nominal variables.

    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "cramers_v"
    symmetric = True

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.categorical(sr_a) or not check.categorical(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
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

        return v


class KolmogorovSmirnovDistance(TwoColumnMetric):
    """Kolmogorov-Smirnov statistic between two continuous variables.

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
            The Kolmogorov-Smirnov distance between sr_a and sr_b.
        """
        column_old_clean = pd.to_numeric(sr_a, errors='coerce').dropna()
        column_new_clean = pd.to_numeric(sr_b, errors='coerce').dropna()
        if len(column_old_clean) == 0 or len(column_new_clean) == 0:
            return np.nan

        ks_distance, p_value = ks_2samp(column_old_clean, column_new_clean)
        return ks_distance


class EarthMoversDistance(TwoColumnMetric):
    """Earth mover's distance (aka 1-Wasserstein distance) between two nominal variables.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "earth_movers_distance"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        if not check.categorical(sr_a) or not check.categorical(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a nominal variable.
            sr_b (pd.Series): values of another nominal variable to compare.

        Returns:
            The earth mover's distance between sr_a and sr_b.
        """
        old = sr_a.to_numpy().astype(str)
        new = sr_b.to_numpy().astype(str)

        space = set(old).union(set(new))
        if len(space) > 1e4:
            return np.nan

        old_unique, counts = np.unique(old, return_counts=True)
        old_counts = dict(zip(old_unique, counts))

        new_unique, counts = np.unique(new, return_counts=True)
        new_counts = dict(zip(new_unique, counts))

        p = np.array([float(old_counts[x]) if x in old_counts else 0.0 for x in space])
        q = np.array([float(new_counts[x]) if x in new_counts else 0.0 for x in space])

        p /= np.sum(p)
        q /= np.sum(q)

        distance = 0.5 * np.sum(np.abs(p.astype(np.float64) - q.astype(np.float64)))
        return distance


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
