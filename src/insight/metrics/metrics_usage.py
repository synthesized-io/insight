from itertools import permutations
from typing import Union

import numpy as np
import pandas as pd

from .base import DataFrameMetric, OneColumnMetric, TwoColumnMetric, TwoDataFrameMetric


class OneColumnMap(DataFrameMetric):
    """
    Mapping of a metric to each column of a dataframe.
    """

    def summarize_result(self, result: pd.DataFrame):
        """
        Give a single value that summarizes the result of the metric. For OneColumnMap it is the mean of the results.

        Args:
            result: the result of the metric computation.
        """
        return result["metric_val"].mean(axis=0)

    def __init__(self, metric: OneColumnMetric):
        self._metric = metric
        self.name = f"{metric.name}_map"

    def _compute_result(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_map = {
            col: self._metric(
                df[col], dataset_name=df.attrs.get("name", "") + f"_{col}"
            )
            for col in df.columns
        }
        result = pd.DataFrame(
            data=columns_map.values(), index=df.columns, columns=["metric_val"]
        )

        result.name = self._metric.name
        return result


class CorrMatrix(DataFrameMetric):
    """Computes the correlation between each pair of columns in the given dataframe
    and returns the result in a dataframe"""

    def summarize_result(self, result: pd.DataFrame):
        """
        Give a single value that summarizes the result of the metric. For CorrMatrix it is the maximum value in the
        matrix.
        Args:
            result: the result of the metric computation.
        """
        return result.values.max()

    def __init__(self, metric: TwoColumnMetric):
        self._metric = metric
        self.name = f"{metric.name}_matrix"

    def _compute_result(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = df.columns
        matrix = pd.DataFrame(index=columns, columns=columns)

        for col_a, col_b in permutations(columns, 2):
            matrix[col_a][col_b] = self._metric(
                df[col_a],
                df[col_b],
                dataset_name=df.attrs.get("name", "") + f"_{col_a}_{col_b}",
            )

        return pd.DataFrame(matrix.astype(np.float32))  # explicit casting for mypy


class DiffCorrMatrix(TwoDataFrameMetric):
    """Computes the correlation matrix for each of the given dataframes and return the difference
    between these matrices"""

    def summarize_result(self, result: pd.DataFrame):
        """
        Give a single value that summarizes the result of the metric. For DiffCorrMatrix it is the maximum absolute
        value in the matrix.
        Args:
            result: the result of the metric computation.
        """
        return result.abs().max().max() # max().max() = max in each col -> max across cols

    def __init__(self, metric: TwoColumnMetric):
        self._corr_matrix = CorrMatrix(metric)
        self.name = f"diff_{metric.name}"

    def _compute_result(
        self, df_old: pd.DataFrame, df_new: pd.DataFrame
    ) -> Union[pd.DataFrame, None]:
        corr_matrix_old = self._corr_matrix(df=df_old)
        corr_matrix_new = self._corr_matrix(df=df_new)

        if corr_matrix_old is None or corr_matrix_new is None:
            return None

        return corr_matrix_old - corr_matrix_new


class TwoColumnMap(TwoDataFrameMetric):
    """Compares columns with the same name from two given dataframes and return a DataFrame
    with index as the column name and the columns as metric_val"""

    def summarize_result(self, result: pd.DataFrame):
        """
        Give a single value that summarizes the result of the metric. For TwoColumnMap it is the mean of the results.
        Args:
            result: the result of the metric computation.
        """
        return result["metric_val"].mean(axis=0)

    def __init__(self, metric: TwoColumnMetric):
        self._metric = metric
        self.name = f"{metric.name}_map"

    def _compute_result(
        self, df_old: pd.DataFrame, df_new: pd.DataFrame
    ) -> pd.DataFrame:
        columns_map = {
            col: self._metric(
                df_old[col],
                df_new[col],
                dataset_name=df_old.attrs.get("name", "") + f"_{col}",
            )
            for col in df_old.columns
        }
        result = pd.DataFrame(
            data=columns_map.values(), index=df_old.columns, columns=["metric_val"]
        )

        result.name = self._metric.name
        return result
