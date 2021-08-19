"""
Collection of Metrics that measure the distance, or similarity, between two datasets.
"""
import warnings

import dcor as dcor
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ..check import ColumnCheck
from .base import MetricStatistics
from .utils import MetricStatisticsResult


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
