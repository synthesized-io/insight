"""This module contains various metrics used across synthesized."""
from typing import Any, Dict, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, kendalltau, wasserstein_distance

from ..check import Check, ColumnCheck
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
        mean = np.nanmean(sr.values - np.array(0, dtype=sr.dtype))
        return mean + np.array(0, dtype=sr.dtype)


class StandardDeviation(OneColumnMetric):
    name = "standard_deviation"

    def __init__(self, check: Check = ColumnCheck(), remove_outliers: float = 0.0):
        super().__init__(check)
        self.remove_outliers = remove_outliers

    def to_dict(self) -> Dict[str, Any]:
        dictionary = super().to_dict()
        dictionary.update({'remove_outliers': self.remove_outliers})
        return dictionary

    @classmethod
    def check_column_types(cls, sr: pd.Series, check: Check = ColumnCheck()):
        if not check.affine(sr):
            return False
        return True

    def _compute_metric(self, sr: pd.Series):
        values = np.sort(sr.values)  # type: ignore
        values = values[int(len(sr) * self.remove_outliers):int(len(sr) * (1.0 - self.remove_outliers))]
        trimmed_sr = pd.Series(values, name=sr.name)

        affine_mean = Mean(upload_to_database=False)
        d = trimmed_sr - affine_mean(trimmed_sr)
        u = d / np.array(1, dtype=d.dtype)
        s = np.sqrt(np.sum(u ** 2))
        return s * np.array(1, dtype=d.dtype)


class KendallTauCorrelation(TwoColumnMetric):
    """Kendall's Tau correlation coefficient between ordinal variables.

    The statistic ranges from -1 to 1, indicating the strength and direction of the relationship between the
    two variables.

    """

    name = "kendall_tau_correlation"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()):
        if len(sr_a) != len(sr_b):
            return False
        if not check.ordinal(sr_a) or not check.ordinal(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of an ordinal variable.
            sr_b (pd.Series): values of another ordinal variable to assess association.

        Returns:
            The Kendall Tau coefficient between sr_a and sr_b.
        """
        if hasattr(sr_a, "cat") and sr_a.cat.ordered:
            sr_a = sr_a.cat.codes

        if hasattr(sr_b, "cat") and sr_b.cat.ordered:
            sr_b = sr_b.cat.codes

        corr, _ = kendalltau(sr_a.values, sr_b.values, nan_policy="omit")

        return corr


class CramersV(TwoColumnMetric):
    """Cramér's V correlation coefficient between nominal variables.

    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e. one variable is completely determined by the other).
    """
    name = "cramers_v"

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

    For order 2 norm, the statistic ranges from 0 to 1 where a value of 0 indicates the two variables follow identical
    distributions, and a value of 1 indicates they follow completely different distributions. The upper bound may change
    depending on the order of the norm chosen.

    Args:
        ord (float, optional):
                The order of the norm. Possible values include real numbers, inf, -inf.
                See numpy.linalg.norm for more details. Defaults to 2.
                Cannot be set to 'fro', or 'nuc' as these are matrix norms.

    Usage:
        Given some Pandas series.
        >>> sr1 = pd.Series([1,2,3,4,5,6,7])
        >>> sr2 = pd.Series([1,2,3,4,5,6,7])

        Create and configure the metric.
        >>> norm = Norm(1)

        Evaluate the metric.
        >>> norm(sr1, sr2)
        0.0
    """
    name = "norm"

    def __init__(self, check: Check = ColumnCheck(), ord: float = 2.0):
        super().__init__(check)
        self.ord = ord

    def to_dict(self) -> Dict[str, Any]:
        dictionary = super().to_dict()
        dictionary.update({'ord': self.ord})
        return dictionary

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
            return np.linalg.norm(cast(pd.Series, p) - cast(pd.Series, q), ord=self.ord)  # type: ignore
        return None


class EarthMoversDistanceBinned(TwoColumnMetric):
    """Earth movers distance (EMD), aka Wasserstein 1-distance, between two histograms.

    The histograms can represent counts of nominal categories or counts on
    an ordinal range. If the latter, they must have equal binning.

    Args:
        bin_edges: Optional; If given, this must be an iterable of bin edges for x and y,
                i.e. the output of np.histogram_bin_edges. If None, then it is assumed
                that the data represent counts of nominal categories, with no meaningful
                distance between bins.

    Usage:

        Nominal:
            Given some Pandas series.
            >>> sr1 = pd.Series([16, 2, 51])
            >>> sr2 = pd.Series([12, 41, 14])

            Create and configure the metric.
            >>> nominal_emd = EarthMoversDistanceBinned()

            Evaluate the metric.
            >>> nominal_emd(sr1, sr2)
            0.5829547912610858

        Ordinal:
            Given some Pandas serieses.
            >>> sr1 = pd.Series([0.73917425, 0.45634101, 0.0769353, 0.1913571, 0.2978581 ,
                ...                  0.76160552, 0.62878134, 0.14740323, 0.19678186, 0.42713395])
            >>> sr2 = pd.Series([0.14313188, 0.23245435, 0.85235284, 0.7497944 , 0.89014916,
                ...                  0.13817053, 0.57767209, 0.0167717 , 0.25390184, 0.62945724])

            Bin the columns.
            >>> bins = np.histogram_bin_edges(pd.concat([sr1, sr2]))

            Create and configure the metric.
            >>> ordinal_emd = EarthMoversDistanceBinned(bin_edges=bins)

            Evaluate the metric.
            >>> ordinal_emd(sr1, sr2)
            0.06876915155978315

    """
    name = "earth_movers_distance_binned"

    def __init__(self,
            check: Check = ColumnCheck(),
            bin_edges: Optional[Union[pd.Series, Sequence, np.ndarray]] = None):
        super().__init__(check)
        self.bin_edges = bin_edges

    def to_dict(self) -> Dict[str, Any]:
        dictionary = super().to_dict()
        dictionary.update({'bin_edges': self.bin_edges})
        return dictionary

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
            # and therefore distances between bins are meaningless. Set to all distances to
            # unity to model this.
            distance = 0.5 * np.sum(np.abs(x.astype(np.float64) - y.astype(np.float64)))
        else:
            # otherwise, use pair-wise euclidean distances between bin centers for scale data
            bin_centers = self.bin_edges[:-1] + np.diff(self.bin_edges) / 2.
            distance = wasserstein_distance(bin_centers, bin_centers, u_weights=x, v_weights=y)
        return distance


class BhattacharyyaCoefficient(TwoColumnMetric):
    """Bhattacharyya coefficient for approximation of overlap between two probability distributions.

    The statistic ranges from 0 to 1, where 1 indicates a complete overlap in the distributions,
    and 0 indicates lack of overlap between the distributions. Bhattacharyya coefficient is closely related to Hellinger
    distance.
    """
    name = "bhattacharyya_coefficient"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()):
        if check.continuous(sr_a) and check.continuous(sr_b):
            return True
        if check.categorical(sr_a) and check.categorical(sr_b):
            return True
        return False

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        (p, q) = zipped_hist((sr_a, sr_b), check=self.check)
        return np.sum(np.sqrt(cast(pd.Series, p) * cast(pd.Series, q)))


class TotalVariationDistance(TwoColumnMetric):
    """Total Variation Distance between probability distributions.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "total_variation_distance"

    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        if check.continuous(sr_a) and check.continuous(sr_b):
            return True
        if check.categorical(sr_a) and check.categorical(sr_b):
            return True
        return False

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        (p, q) = zipped_hist((sr_a, sr_b), check=self.check)
        return np.linalg.norm(cast(pd.Series, p) - cast(pd.Series, q), ord=1) / 2
