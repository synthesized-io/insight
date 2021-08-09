import warnings
from typing import Union

import dcor as dcor
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scipy.stats import kendalltau, ks_2samp, spearmanr, entropy, kruskal
from scipy.spatial.distance import jensenshannon

from .base import OneColumnMetric, TwoColumnMetric
from src.synthesized_insight import ColumnCheck
from .utils import zipped_hist, infer_distr_type


class Mean(OneColumnMetric):
    name = "mean"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr: pd.Series):
        print(check.affine(sr))
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
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        if not check.categorical(sr_a) or not check.continuous(sr_b):
            return False
        return True

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a nominal variable.
            sr_b (pd.Series): values of numerical variable.

        Returns:
            The R2 Mcfadden correlation coefficient between sr_a and sr_b.
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

        return pseudo_r2


class DistanceNNCorrelation(TwoColumnMetric):
    """Distance nn correlation coefficient between two numerical variables.

    It uses non-linear correlation distance to obtain a correlation coefficient for
    numerical-numerical column pairs.

    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "distance_nn_correlation"

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
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
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
        warnings.filterwarnings(action="ignore", category=UserWarning)

        sr_a = sr_a.astype("category").cat.codes
        groups = sr_b.groupby(sr_a)
        arrays = [groups.get_group(category) for category in sr_a.unique()]

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

        total /= n * (n - 1) / 2

        if total is None:
            return 0.0

        return total


class BinomialDistance(TwoColumnMetric):
    """Binomial distance between two binary variables.

    The statistic ranges from 0 to 1, where a value of 0 indicates there is no association between the variables,
    and 1 indicates maximal association (i.e one variable is completely determined by the other).
    """
    name = "binomial_distance"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        return infer_distr_type(pd.concat((sr_a, sr_b))).is_binary()

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a numerical variable.
            sr_b (pd.Series): values of numerical variable.

        Returns:
            The Binomial distance between sr_a and sr_b.
        """
        return sr_a.mean() - sr_b.mean()


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


class KruskalWallis(TwoColumnMetric):
    """Kruskal Wallis distance between two numerical variables.

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
        return kruskal(sr_a, sr_b)[0]


class KullbackLeiblerDivergence(TwoColumnMetric):
    """Kullback–Leibler Divergence or Relative Entropy between two probability distributions.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "kullback_leibler_divergence"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        x_dtype = str(check.infer_dtype(sr_a).dtype)
        y_dtype = str(check.infer_dtype(sr_b).dtype)

        return x_dtype == y_dtype

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            The kullback-leibler divergence between sr_a and sr_b.
        """
        (p, q), _ = zipped_hist((sr_a, sr_b), ret_bins=True)
        return entropy(np.array(p), np.array(q))


class JensenShannonDivergence(TwoColumnMetric):
    """Jensen-Shannon Divergence between two probability distributions.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "jensen_shannon_divergence"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        x_dtype = str(check.infer_dtype(sr_a).dtype)
        y_dtype = str(check.infer_dtype(sr_b).dtype)

        return x_dtype == y_dtype

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            The jensen-shannon divergence between sr_a and sr_b.
        """
        (p, q), _ = zipped_hist((sr_a, sr_b), ret_bins=True)
        return jensenshannon(p, q)


class HellingerDistance(TwoColumnMetric):
    """Hellinger Distance between two probability distributions.

    The statistic ranges from 0 to 1, where a value of 0 indicates the two variables follow identical distributions,
    and a value of 1 indicates they follow completely different distributions.
    """
    name = "hellinger_distance"

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        x_dtype = str(check.infer_dtype(sr_a).dtype)
        y_dtype = str(check.infer_dtype(sr_b).dtype)

        return x_dtype == y_dtype

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            The hellinger distance between sr_a and sr_b.
        """
        (p, q), _ = zipped_hist((sr_a, sr_b), ret_bins=True)
        return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)


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

    def __init__(self, check: ColumnCheck = None, ord: Union[str, int] = 2):
        super().__init__(check)
        self.ord = ord

    @classmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series) -> bool:
        x_dtype = str(check.infer_dtype(sr_a).dtype)
        y_dtype = str(check.infer_dtype(sr_b).dtype)

        return x_dtype == y_dtype

    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        """Calculate the metric.

        Args:
            sr_a (pd.Series): values of a variable.
            sr_b (pd.Series): values of another variable to compare.

        Returns:
            The lp-norm between sr_a and sr_b.
        """
        (p, q), _ = zipped_hist((sr_a, sr_b), ret_bins=True)
        return np.linalg.norm(p - q, ord=self.ord)


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
