"""This module contains various metrics used across synthesized."""
import logging
from typing import List, Union

import numpy as np
import pandas as pd
from pyemd import emd
from scipy.stats import kendalltau, ks_2samp, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from .metrics_base import ColumnMetric, TwoColumnMetric
from ..modelling import ModellingPreprocessor
from ...metadata import Affine, Ordinal
from ...metadata.factory import MetaExtractor
from ...model import ContinuousModel, DataFrameModel, DiscreteModel
from ...model.factory import ModelFactory

logger = logging.getLogger(__name__)


class Mean(ColumnMetric):
    name = "mean"

    def check_column_types(self, sr_a: pd.Series, df_model: DataFrameModel = None):
        df_model = self.extract_models(sr_a, df_model)

        if not isinstance(df_model[sr_a.name].meta, Affine):
            return False
        return True

    def __call__(self, sr: pd.Series, df_model: DataFrameModel = None) -> Union[float, None]:
        if not self.check_column_types(sr, df_model=df_model):
            return None

        return affine_mean(sr)


class StandardDeviation(ColumnMetric):
    name = "standard_deviation"

    def __init__(self, remove_outliers: float = 0.0):
        self.remove_outliers = remove_outliers
        super().__init__()

    def check_column_types(self, sr_a: pd.Series, df_model: DataFrameModel = None):
        df_model = self.extract_models(sr_a, df_model)

        if not isinstance(df_model[sr_a.name].meta, Affine):
            return False
        return True

    def __call__(self, sr: pd.Series, df_model: DataFrameModel = None) -> Union[int, float, None]:
        if not self.check_column_types(sr, df_model=df_model):
            return None

        values = np.sort(sr.values)[int(len(sr) * self.remove_outliers):int(len(sr) * (1.0 - self.remove_outliers))]
        stddev = affine_stddev(pd.Series(values, name=sr.name))

        return stddev


class KendellTauCorrelation(TwoColumnMetric):
    name = "kendell_tau_correlation"
    symmetric = True

    def __init__(self, max_p_value: float = 1.0, calculate_categorical: bool = False):
        self.max_p_value = max_p_value
        self.calculate_categorical = calculate_categorical
        super().__init__()

    def check_column_types(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None):
        df_model = self.extract_models(sr_a, sr_b, df_model)
        if not self.calculate_categorical and (not isinstance(df_model[sr_a.name], ContinuousModel)
                                               or not isinstance(df_model[sr_b.name], ContinuousModel)):
            return False

        if not isinstance(df_model[sr_a.name].meta, Ordinal) or not isinstance(df_model[sr_b.name].meta, Ordinal):
            return False
        return True

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> Union[int, float, None]:
        sr_a = pd.to_numeric(sr_a, errors='coerce')
        sr_b = pd.to_numeric(sr_b, errors='coerce')

        if not self.check_column_types(sr_a, sr_b, df_model=df_model):
            return None

        corr, p_value = kendalltau(sr_a.values, sr_b.values, nan_policy='omit')

        if p_value <= self.max_p_value:
            return corr
        else:
            return None


class SpearmanRhoCorrelation(TwoColumnMetric):
    name = "spearman_rho_correlation"

    def __init__(self, max_p_value: float = 1.0):
        self.max_p_value = max_p_value
        super().__init__()

    def check_column_types(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None):
        df_model = self.extract_models(sr_a, sr_b, df_model=df_model)

        if not isinstance(df_model[sr_a.name].meta, Ordinal) or not isinstance(df_model[sr_b.name].meta, Ordinal):
            return False
        return True

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> Union[int, float, None]:
        if not self.check_column_types(sr_a, sr_b, df_model=df_model):
            return None

        corr, p_value = spearmanr(sr_a.values, sr_b.values)

        if p_value <= self.max_p_value:
            return corr
        else:
            return None


class CramersV(TwoColumnMetric):
    name = "cramers_v"
    symmetric = True

    def check_column_types(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> bool:
        df_model = self.extract_models(sr_a, sr_b, df_model)

        if not isinstance(df_model[sr_a.name], DiscreteModel) or not isinstance(df_model[sr_b.name], DiscreteModel):
            return False
        return True

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> Union[int, float, None]:
        if not self.check_column_types(sr_a, sr_b, df_model=df_model):
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

    def check_column_types(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> bool:
        df_model = self.extract_models(sr_a, sr_b, df_model=df_model)

        x_model = df_model[sr_a.name]
        y_model = df_model[sr_b.name]

        if not isinstance(x_model, ContinuousModel) or\
           (isinstance(x_model, ContinuousModel) and x_model.meta.dtype == 'M8[ns]'):
            return False
        if not isinstance(y_model, DiscreteModel):
            return False
        return True

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> Union[int, float, None]:
        if not self.check_column_types(sr_a, sr_b, df_model=df_model):
            return None

        df = pd.DataFrame(data={sr_a.name: sr_a, sr_b.name: sr_b})
        r2 = logistic_regression_r2(df, y_label=sr_b.name, x_labels=[sr_a.name], df_model=df_model)

        return r2


class KolmogorovSmirnovDistance(TwoColumnMetric):
    name = "kolmogorov_smirnov_distance"

    def check_column_types(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> bool:
        df_model = self.extract_models(sr_a, sr_b, df_model=df_model)

        if not isinstance(df_model[sr_a.name], ContinuousModel) and not isinstance(df_model[sr_b.name], ContinuousModel):
            return False
        return True

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> Union[int, float, None]:
        if not self.check_column_types(sr_a, sr_b, df_model=df_model):
            return None
        column_old_clean = pd.to_numeric(sr_a, errors='coerce').dropna()
        column_new_clean = pd.to_numeric(sr_b, errors='coerce').dropna()
        if len(column_old_clean) == 0 or len(column_new_clean) == 0:
            return np.nan

        ks_distance, p_value = ks_2samp(column_old_clean, column_new_clean)
        return ks_distance


class EarthMoversDistance(TwoColumnMetric):
    name = "earth_movers_distance"

    def check_column_types(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> bool:
        df_model = self.extract_models(sr_a, sr_b, df_model=df_model)

        if not isinstance(df_model[sr_a.name], DiscreteModel) and not isinstance(df_model[sr_b.name], DiscreteModel):
            return False
        return True

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, df_model: DataFrameModel = None) -> Union[int, float, None]:
        if not self.check_column_types(sr_a, sr_b, df_model=df_model):
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


def logistic_regression_r2(
        df: pd.DataFrame, y_label: str, x_labels: List[str], max_sample_size: int = 10_000,
        df_model: DataFrameModel = None
) -> Union[None, float]:
    if df_model is None:
        df_meta = MetaExtractor.extract(df=df)
        df_model = ModelFactory()(df_meta)

    if not isinstance(df_model[y_label], DiscreteModel):
        return None

    if len(x_labels) == 0:
        return None

    df = df[x_labels + [y_label]].dropna()
    df = df.sample(min(max_sample_size, len(df)))

    if df[y_label].nunique() < 2:
        return None

    df_pre = ModellingPreprocessor.preprocess(data=df, target=y_label, df_model=df_model)
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
