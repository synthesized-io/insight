"""This module contains various metrics used across synthesized."""
from typing import List, Union

import numpy as np
import pandas as pd
from pyemd import emd
from scipy.stats import kendalltau, spearmanr, ks_2samp

from .metrics_base import ColumnMetric, TwoColumnMetric, DataFrameMetric, ColumnComparison


class StandardDeviation(ColumnMetric):
    name = "Standard Deviation"
    tags = ["ordinal"]

    @staticmethod
    def compute(df: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        column = df[col_name]
        stddev = float(np.var(column.values)**0.5)

        return stddev


class KendellTauCorrelation(TwoColumnMetric):
    name = "Kendell's Tau correlation"
    tags = ["ordinal", "symmetric"]

    @staticmethod
    def compute(df: pd.DataFrame, col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        column_a = df[col_a_name]
        column_b = df[col_b_name]
        corr, p_value = kendalltau(column_a.values, column_b.values)

        return corr


class SpearmanRhoCorrelation(TwoColumnMetric):
    name = "Spearman's Rho correlation"
    tags = ["ordinal", "symmetric"]

    @staticmethod
    def compute(df: pd.DataFrame, col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        column_a = df[col_a_name]
        column_b = df[col_b_name]
        corr, p_value = spearmanr(column_a.values, column_b.values)

        return corr


class CramersV(TwoColumnMetric):
    name = "Cramer's V"
    tags = ["nominal", "symmetric"]

    @staticmethod
    def compute(df: pd.DataFrame, col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        column_a = df[col_a_name]
        column_b = df[col_b_name]
        table = pd.crosstab(column_a, column_b)
        expected = table.fittedvalues.to_numpy()
        real = table.table
        r, c = real.shape
        n = np.sum(real)
        v = np.sum((real - expected) ** 2 / (expected * n * min(r - 1, c - 1))) ** 0.5

        return v


class KolmogorovSmirnovDistance(ColumnComparison):
    name = "KS Distance"
    tags = ["continuous"]

    @staticmethod
    def compute(df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        column_old_clean = df_old[col_name].dropna()
        column_new_clean = df_new[col_name].dropna()
        ks_distance, p_value = ks_2samp(column_old_clean, column_new_clean)
        return ks_distance


class EarthMoversDistance(ColumnComparison):
    name = "EM Distance"
    tags = ["categorical"]

    @staticmethod
    def compute(df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
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
