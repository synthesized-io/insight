from itertools import permutations
from typing import Union

import numpy as np
import pandas as pd

from .base import DataFrameMetric, OneColumnMetric, TwoColumnMetric, TwoDataFrameMetric


class OneColumnMap(DataFrameMetric):
    def __init__(self, metric: OneColumnMetric):
        self._metric = metric
        self.name = f'{metric.name}_map'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_map = {col: self._metric(df[col]) for col in df.columns}
        result = pd.DataFrame(
            data=columns_map.values(),
            index=df.columns,
            columns=['metric_val']
        )

        result.name = self._metric.name
        return result


class CorrMatrix(DataFrameMetric):
    """Computes the correlation between each pair of columns in the given dataframe
    and returns the result in a dataframe"""

    def __init__(self, metric: TwoColumnMetric):
        self._metric = metric
        self.name = f'{metric.name}_matrix'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = df.columns
        matrix = pd.DataFrame(index=columns, columns=columns)

        for col_a, col_b in permutations(columns, 2):
            matrix[col_a][col_b] = self._metric(df[col_a], df[col_b])

        return pd.DataFrame(matrix.astype(np.float32))  # explicit casting for mypy


class DiffCorrMatrix(TwoDataFrameMetric):
    """Computes the correlation matrix for each of the given dataframes and return the difference
    between these matrices"""

    def __init__(self, metric: TwoColumnMetric):
        self._corr_matrix = CorrMatrix(metric)
        self.name = f'diff_{metric.name}'

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[pd.DataFrame, None]:
        corr_matrix_old = self._corr_matrix(df=df_old)
        corr_matrix_new = self._corr_matrix(df=df_new)

        if corr_matrix_old is None or corr_matrix_new is None:
            return None

        return corr_matrix_old - corr_matrix_new


class TwoColumnMap(TwoDataFrameMetric):
    """Compares columns with the same name from two given dataframes and return a DataFrame
    with index as the column name and the columns as metric_val"""

    def __init__(self, metric: TwoColumnMetric):
        self._metric = metric
        self.name = f'{metric.name}_map'

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
        columns_map = {col: self._metric(df_old[col], df_new[col]) for col in df_old.columns}
        result = pd.DataFrame(
            data=columns_map.values(),
            index=df_old.columns,
            columns=['metric_val']
        )

        result.name = self._metric.name
        return result
