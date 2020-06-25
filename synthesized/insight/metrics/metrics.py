"""This module contains various metrics used across synthesized."""
from typing import List, Union

import numpy as np
import pandas as pd
from pyemd import emd
import statsmodels.api as sm
from scipy.stats import kendalltau, spearmanr, ks_2samp

from .metrics_base import ColumnMetric, TwoColumnMetric, DataFrameMetric, ColumnComparison, DataFrameComparison
from ..modelling import predictive_modelling_score, predictive_modelling_comparison, logistic_regression_r2


class Mean(ColumnMetric):
    name = "mean"
    tags = ["ordinal"]

    def __call__(self, sr: pd.Series, **kwargs) -> float:
        mean = float(np.nanmean(sr.values))

        return mean


class StandardDeviation(ColumnMetric):
    name = "standard_deviation"
    tags = ["ordinal"]

    def __call__(self, sr: pd.Series, **kwargs) -> Union[int, float, None]:
        stddev = float(np.var(sr.values)**0.5)

        return stddev


class KendellTauCorrelation(TwoColumnMetric):
    name = "kendell_tau_correlation"
    tags = ["ordinal", "symmetric"]

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, **kwargs) -> Union[int, float, None]:
        if not super().check_column_types(sr_a, sr_b, **kwargs):
            return None

        corr, p_value = kendalltau(sr_a.values, sr_b.values)

        if p_value <= kwargs.get('max_p_value', 1.0):
            return corr
        else:
            return None


class SpearmanRhoCorrelation(TwoColumnMetric):
    name = "spearman_rho_correlation"
    tags = ["ordinal", "symmetric"]

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, **kwargs) -> Union[int, float, None]:
        if not super().check_column_types(sr_a, sr_b, **kwargs):
            return None

        corr, p_value = spearmanr(sr_a.values, sr_b.values)

        if p_value <= kwargs.get('max_p_value', 1.0):
            return corr
        else:
            return None


class CramersV(TwoColumnMetric):
    name = "cramers_v"
    tags = ["nominal", "symmetric"]

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, **kwargs) -> Union[int, float, None]:
        if not super().check_column_types(sr_a, sr_b, **kwargs):
            return None

        table = sm.stats.Table(pd.crosstab(sr_a, sr_b))
        expected = table.fittedvalues.to_numpy()
        real = table.table
        r, c = real.shape
        n = np.sum(real)
        v = np.sum((real - expected) ** 2 / (expected * n * min(r - 1, c - 1))) ** 0.5

        return v


class CategoricalLogisticR2(TwoColumnMetric):
    name = "categorical_logistic_correlation"

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, **kwargs) -> Union[int, float, None]:
        if not super().check_column_types(sr_a, sr_b, **kwargs):
            return None

        df = pd.DataFrame(data=[sr_a, sr_b])
        r2 = logistic_regression_r2(df, y_label=sr_b.name, x_labels=[sr_a.name], **kwargs)

        return r2


class KolmogorovSmirnovDistance(ColumnComparison):
    name = "kolmogorov_smirnov_distance"
    tags = ["ordinal"]

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        if not super(KolmogorovSmirnovDistance, self).check_column_types(df_old, df_new, col_name, **kwargs):
            return None
        column_old_clean = df_old[col_name].dropna()
        column_new_clean = df_new[col_name].dropna()
        ks_distance, p_value = ks_2samp(column_old_clean, column_new_clean)
        return ks_distance


class EarthMoversDistance(ColumnComparison):
    name = "earth_movers_distance"
    tags = ["nominal"]

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        if not super(EarthMoversDistance, self).check_column_types(df_old, df_new, col_name, **kwargs):
            return None

        old = df_old[col_name].to_numpy()
        new = df_new[col_name].to_numpy()

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

        distances = 1 - np.eye(len(space))

        return emd(p, q, distances)


class PredictiveModellingScore(DataFrameMetric):
    name = "predictive_modelling_score"
    tags = ["modelling"]

    def __call__(self, df: pd.DataFrame, model: str = None, y_label: str = None,
                 x_labels: List[str] = None, **kwargs) -> Union[int, float, None]:
        if len(df.columns) < 2:
            raise ValueError
        model = model or 'Linear'
        y_label = y_label or df.columns[-1]
        x_labels = x_labels if x_labels is not None else [col for col in df.columns if col != y_label]

        score, metric, task = predictive_modelling_score(df, y_label, x_labels, model)
        return score


class PredictiveModellingComparison(DataFrameComparison):
    name = "predictive_modelling_comparison"
    tags = ["modelling"]

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, model: str = None, y_label: str = None,
                 x_labels: List[str] = None, **kwargs) -> Union[None, float]:
        if len(df_old.columns) < 2:
            raise ValueError
        model = model or 'Linear'
        y_label = y_label or df_old.columns[-1]
        x_labels = x_labels if x_labels is not None else [col for col in df_old.columns if col != y_label]

        score, synth_score, metric, task = predictive_modelling_comparison(df_old, df_new, y_label, x_labels, model)
        return synth_score/score
