"""This module contains various metrics used across synthesized."""
import logging
from typing import List, Union

import numpy as np
import pandas as pd
from pyemd import emd
from scipy.stats import kendalltau, spearmanr, ks_2samp
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from .metrics_base import ColumnMetric, TwoColumnMetric
from ..modelling import ModellingPreprocessor
from ...metadata import MetaExtractor

logger = logging.getLogger(__name__)


class Mean(ColumnMetric):
    name = "mean"
    tags = ["ordinal"]

    def __call__(self, sr: pd.Series = None, **kwargs) -> Union[float, None]:
        if sr is None:
            return None

        mean = float(np.nanmean(sr.values))

        return mean


class StandardDeviation(ColumnMetric):
    name = "standard_deviation"
    tags = ["ordinal"]

    def __call__(self, sr: pd.Series = None, **kwargs) -> Union[int, float, None]:
        if sr is None:
            return None

        rm_outliers = kwargs.get('rm_outliers', 0.0)
        values = np.sort(sr.values)[int(len(sr) * rm_outliers):int(len(sr) * (1.0 - rm_outliers))]
        stddev = float(np.nanvar(values)**0.5)

        return stddev


class KendellTauCorrelation(TwoColumnMetric):
    name = "kendell_tau_correlation"
    tags = ["ordinal", "symmetric"]

    def __call__(self, sr_a: pd.Series = None, sr_b: pd.Series = None, **kwargs) -> Union[int, float, None]:
        if sr_a is None or sr_b is None:
            return None

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

    def __call__(self, sr_a: pd.Series = None, sr_b: pd.Series = None, **kwargs) -> Union[int, float, None]:
        if sr_a is None or sr_b is None:
            return None

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

    def __call__(self, sr_a: pd.Series = None, sr_b: pd.Series = None, **kwargs) -> Union[int, float, None]:
        if sr_a is None or sr_b is None:
            return None

        if not super().check_column_types(sr_a, sr_b, **kwargs):
            return None

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


class CategoricalLogisticR2(TwoColumnMetric):
    name = "categorical_logistic_correlation"
    tags = ["continuous_input_only", "categorical_output_only"]

    def __call__(self, sr_a: pd.Series = None, sr_b: pd.Series = None, **kwargs) -> Union[int, float, None]:
        if sr_a is None or sr_b is None:
            return None

        if not super().check_column_types(sr_a, sr_b, **kwargs):
            return None

        df = pd.DataFrame(data={sr_a.name: sr_a, sr_b.name: sr_b})
        r2 = logistic_regression_r2(df, y_label=sr_b.name, x_labels=[sr_a.name], **kwargs)

        return r2


class KolmogorovSmirnovDistance(TwoColumnMetric):
    name = "kolmogorov_smirnov_distance"
    tags = ["ordinal"]

    def __call__(self, sr_a: pd.Series = None, sr_b: pd.Series = None, **kwargs) -> Union[int, float, None]:
        if sr_a is None or sr_b is None:
            return None

        if not super(KolmogorovSmirnovDistance, self).check_column_types(sr_a, sr_b, **kwargs):
            return None
        column_old_clean = pd.to_numeric(sr_a, errors='coerce').dropna()
        column_new_clean = pd.to_numeric(sr_b, errors='coerce').dropna()
        if len(column_old_clean) == 0 or len(column_new_clean) == 0:
            return np.nan

        ks_distance, p_value = ks_2samp(column_old_clean, column_new_clean)
        return ks_distance


class EarthMoversDistance(TwoColumnMetric):
    name = "earth_movers_distance"
    tags = ["nominal"]

    def __call__(self, sr_a: pd.Series = None, sr_b: pd.Series = None, **kwargs) -> Union[int, float, None]:
        if sr_a is None or sr_b is None:
            return None

        if not super(EarthMoversDistance, self).check_column_types(sr_a, sr_b, **kwargs):
            return None

        old = sr_a.to_numpy()
        new = sr_b.to_numpy()

        space = set(old).union(set(new))
        if len(space) > 1e4:
            return np.nan
        try:
            old_unique, counts = np.unique(old, return_counts=True)
        except TypeError:
            old_unique, counts = np.unique(old.astype(str), return_counts=True)

        old_counts = dict(zip(old_unique, counts))

        try:
            new_unique, counts = np.unique(new, return_counts=True)
        except TypeError:
            new_unique, counts = np.unique(new.astype(str), return_counts=True)

        new_counts = dict(zip(new_unique, counts))

        p = np.array([float(old_counts[x]) if x in old_counts else 0.0 for x in space])
        q = np.array([float(new_counts[x]) if x in new_counts else 0.0 for x in space])

        p /= np.sum(p)
        q /= np.sum(q)

        distances = 1 - np.eye(len(space))

        return emd(p, q, distances)


def logistic_regression_r2(df: pd.DataFrame, y_label: str, x_labels: List[str],
                           max_sample_size: int = 10_000, **kwargs) -> Union[None, float]:
    dp = kwargs.get('vf')
    if dp is None:
        dp = MetaExtractor.extract(df=df)
    categorical, continuous = dp.get_categorical_and_continuous()
    if y_label not in [v.name for v in categorical]:
        return None
    if len(x_labels) == 0:
        return None

    df = df[x_labels + [y_label]].dropna()
    df = df.sample(min(max_sample_size, len(df)))

    if df[y_label].nunique() < 2:
        return None

    df_pre = ModellingPreprocessor.preprocess(df, target=y_label, dp=dp)
    x_labels_pre = list(filter(lambda v: v != y_label, df_pre.columns))

    x_array = df_pre[x_labels_pre].to_numpy()
    y_array = df[y_label].to_numpy()

    rg = LogisticRegression()
    rg.fit(x_array, y_array)

    labels = df[y_label].map({c: n for n, c in enumerate(rg.classes_)}).to_numpy()
    oh_labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))

    lp = rg.predict_log_proba(x_array)
    llf = np.sum(oh_labels * lp)

    rg = LogisticRegression()
    rg.fit(np.ones_like(y_array).reshape(-1, 1), y_array)

    lp = rg.predict_log_proba(x_array)
    llnull = np.sum(oh_labels * lp)

    psuedo_r2 = 1 - (llf / llnull)

    return psuedo_r2
