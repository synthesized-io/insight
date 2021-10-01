"""This module contains various metrics used across synthesized."""
import warnings
from typing import List, Optional, Sequence, Union, cast

import dcor as dcor
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from ..check import Check, ColumnCheck
from ..modelling import ModellingPreprocessor
from .base import OneColumnMetric, TwoColumnMetric
from .utils import zipped_hist


class Mean(OneColumnMetric):
    name = "mean"

    @classmethod
    def check_column_types(cls, sr: pd.Series, check: Check = ColumnCheck()):
        if not check.affine(sr):
            return False
        return True

    def _compute_metric(self, sr: pd.Series):
        return affine_mean(sr)


class StandardDeviation(OneColumnMetric):
    name = "standard_deviation"

    def __init__(self, check: Check = ColumnCheck(), remove_outliers: float = 0.0):
        super().__init__(check)
        self.remove_outliers = remove_outliers

    @classmethod
    def check_column_types(cls, sr: pd.Series, check: Check = ColumnCheck()):
        if not check.affine(sr):
            return False
        return True

    def _compute_metric(self, sr: pd.Series):
        values = np.sort(sr.values)
        values = values[int(len(sr) * self.remove_outliers):int(len(sr) * (1.0 - self.remove_outliers))]
        return affine_stddev(pd.Series(values, name=sr.name))


class CramersV(TwoColumnMetric):
    """Cramér's V correlation coefficient between nominal variables.

    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "cramers_v"
    symmetric = True

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()):
        if not check.categorical(sr_a) or not check.categorical(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        if len(sr_a.value_counts()) == 1 or len(sr_b.value_counts()) == 1:
            return 0

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


class R2Mcfadden(TwoColumnMetric):
    """R2 Mcfadden correlation coefficient between catgorical and numerical variables.

    It trains two multinomial logistic regression models on the data, one using the numerical
    series as the feature and the other only using the intercept term as the input.
    The categorical column is used for the target labels. It then calculates the null
    and the model likelihoods based on them, which are used to compute the pseudo-R2 McFadden score,
    which is used as a correlation coefficient.
    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "r2_mcfadden"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()):
        if not check.categorical(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _logistic_regression_r2(self,
                                df: pd.DataFrame,
                                y_label: str,
                                x_labels: List[str],
                                max_sample_size: int = 10_000) -> Union[None, float]:

        if len(x_labels) == 0:
            return None

        df = df[x_labels + [y_label]].dropna()
        df = pd.DataFrame(df.sample(min(max_sample_size, len(df))))  # explicit assignment for mypy

        if df[y_label].nunique() < 2:
            return None

        df_pre = ModellingPreprocessor.preprocess(df=df, target=y_label)
        x_labels_pre = list(filter(lambda v: v != y_label, df_pre.columns))

        x_array = df_pre[x_labels_pre].to_numpy()
        y_array = df_pre[y_label].to_numpy()

        rg = LogisticRegression()
        rg.fit(x_array, y_array)

        labels = df_pre[y_label].map({c: n for n, c in enumerate(rg.classes_)}).to_numpy()
        oh_labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))

        lp = rg.predict_log_proba(x_array)
        llf = np.sum(oh_labels * lp)

        rg = LogisticRegression()
        rg.fit(np.ones_like(y_array).reshape(-1, 1), y_array)

        lp = rg.predict_log_proba(x_array)
        llnull = np.sum(oh_labels * lp)

        psuedo_r2 = 1 - (llf / llnull)

        return psuedo_r2

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.
        Args:
            sr_a (pd.Series): values of a nominal variable.
            sr_b (pd.Series): values of numerical variable.
        Returns:
            The R2 Mcfadden correlation coefficient between sr_a and sr_b.
        """
        df = pd.DataFrame(data={sr_a.name: sr_a, sr_b.name: sr_b})
        r2 = self._logistic_regression_r2(df, y_label=str(sr_a.name), x_labels=[str(sr_b.name)])

        return r2


class DistanceNNCorrelation(TwoColumnMetric):
    """Distance nn correlation coefficient between two numerical variables.

    It uses non-linear correlation distance to obtain a correlation coefficient for
    numerical-numerical column pairs.
    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "distance_nn_correlation"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()):
        if not check.continuous(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.
        Args:
            sr_a (pd.Series): values of a numerical variable.
            sr_b (pd.Series): values of numerical variable.
        Returns:
            The distance nn correlation coefficient between sr_a and sr_b.
        """
        warnings.filterwarnings(action="ignore", category=UserWarning)

        if sr_a.size < sr_b.size:
            sr_a = sr_a.append(pd.Series(sr_a.mean()).repeat(sr_b.size - sr_a.size), ignore_index=True)
        elif sr_a.size > sr_b.size:
            sr_b = sr_b.append(pd.Series(sr_b.mean()).repeat(sr_a.size - sr_b.size), ignore_index=True)

        return dcor.distance_correlation(sr_a, sr_b)


class DistanceCNCorrelation(TwoColumnMetric):
    """Distance cn correlation coefficient between categorical and numerical variables.

    It uses non-linear correlation distance to obtain a correlation coefficient for
    categorical-numerical column pairs.
    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "distance_cn_correlation"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()):
        if not check.categorical(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.
        Args:
            sr_a (pd.Series): values of a categorical variable.
            sr_b (pd.Series): values of numerical variable.
        Returns:
            The distance cn correlation coefficient between sr_a and sr_b.
        """
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

        return total


class EarthMoversDistance(TwoColumnMetric):
    """Earth mover's distance (aka 1-Wasserstein distance) between two nominal variables.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "earth_movers_distance"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
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


class KullbackLeiblerDivergence(TwoColumnMetric):
    """Kullback–Leibler Divergence or Relative Entropy between two probability distributions.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "kullback_leibler_divergence"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        if check.continuous(sr_a) and check.continuous(sr_b):
            return True
        if check.categorical(sr_a) and check.categorical(sr_b):
            return True
        return False

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.
        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.
        Returns:
            The kullback-leibler divergence between sr_a and sr_b.
        """
        (p, q) = zipped_hist((sr_a, sr_b), check=self.check)
        return entropy(np.array(p), np.array(q))


class JensenShannonDivergence(TwoColumnMetric):
    """Jensen-Shannon Divergence between two probability distributions.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "jensen_shannon_divergence"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        if check.continuous(sr_a) and check.continuous(sr_b):
            return True
        if check.categorical(sr_a) and check.categorical(sr_b):
            return True
        return False

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.
        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.
        Returns:
            The jensen-shannon divergence between sr_a and sr_b.
        """
        (p, q) = zipped_hist((sr_a, sr_b), check=self.check)
        return jensenshannon(p, q)


class HellingerDistance(TwoColumnMetric):
    """Hellinger Distance between two probability distributions.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "hellinger_distance"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        if check.continuous(sr_a) and check.continuous(sr_b):
            return True
        if check.categorical(sr_a) and check.categorical(sr_b):
            return True
        return False

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.
        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.
        Returns:
            The hellinger distance between sr_a and sr_b.
        """
        (p, q) = zipped_hist((sr_a, sr_b), check=self.check)
        return np.linalg.norm(np.sqrt(cast(pd.Series, p)) - np.sqrt(cast(pd.Series, q))) / np.sqrt(2)


class Norm(TwoColumnMetric):
    """Norm between two probability distributions.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.

    Args:
        ord (Union[str, int], optional):
                The order of the norm. Possible values include positive numbers, 'fro', 'nuc'.
                See numpy.linalg.norm for more details. Defaults to 2.
    """
    name = "norm"

    def __init__(self, check: Check = ColumnCheck(), ord: Union[str, int] = 2):
        super().__init__(check)
        self.ord = ord

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        if check.continuous(sr_a) and check.continuous(sr_b):
            return True
        if check.categorical(sr_a) and check.categorical(sr_b):
            return True
        return False

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.
        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.
        Returns:
            The lp-norm between sr_a and sr_b.
        """
        (p, q) = zipped_hist((sr_a, sr_b), check=self.check)
        if p is not None and q is not None:
            return np.linalg.norm(cast(pd.Series, p) - cast(pd.Series, q), ord=self.ord)
        return None


class EarthMoversDistanceBinned(TwoColumnMetric):
    """
    Earth movers distance (EMD), aka Wasserstein 1-distance, between two histograms.

    The histograms can represent counts of nominal categories or counts on
    an ordinal range. If the latter, they must have equal binning.

    Args:
        bin_edges: Optional; If given, this must be an iterable of bin edges for x and y,
                i.e the output of np.histogram_bin_edges. If None, then it is assumed
                that the data represent counts of nominal categories, with no meaningful
                distance between bins.

    """
    name = "earth_movers_distance_binned"

    def __init__(self,
                 check: Check = ColumnCheck(),
                 bin_edges: Optional[Union[pd.Series, Sequence, np.ndarray]] = None):
        super().__init__(check)
        self.bin_edges = bin_edges

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        # Histograms can appear to be continuous even if they are categorical in nature
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """
        Calculate the EMD between two binned continuous data or two 1d histograms.

        Histograms must have an equal number of bins. They are not required to be normalised,
        and distances between bins are measured using a Euclidean metric.

        Returns:
            The earth mover's distance.
        """
        if sr_a.sum() == 0 and sr_b.sum() == 0:
            return 0.
        elif sr_a.sum() == 0 or sr_b.sum() == 0:
            return 1.

        # normalise counts for consistency with scipy.stats.wasserstein
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(sr_a / sr_a.sum())
            y = np.nan_to_num(sr_b / sr_b.sum())

        if self.bin_edges is None:
            # if bins not given, histograms are assumed to be counts of nominal categories,
            # and therefore distances betwen bins are meaningless. Set to all distances to
            # unity to model this.
            distance = 0.5 * np.sum(np.abs(x.astype(np.float64) - y.astype(np.float64)))
        else:
            # otherwise, use pair-wise euclidean distances between bin centers for scale data
            bin_centers = self.bin_edges[:-1] + np.diff(self.bin_edges) / 2.
            distance = wasserstein_distance(bin_centers, bin_centers, u_weights=x, v_weights=y)
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
