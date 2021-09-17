
from itertools import permutations
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import TwoColumnMetric, TwoColumnTest


class TwoColumnMap:
    """Compares columns with the same name from two given dataframes and return a DataFrame
    with index as the column name and the columns as metric_val and metric_pval(if applicable)"""

    def __init__(self, metric: Union[TwoColumnMetric, TwoColumnTest]):
        self.metric = metric
        self.name = f'{metric.name}_map'

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
        columns_map = {col: self.metric(df_old[col], df_new[col]) for col in df_old.columns}

        result = pd.DataFrame(
            data=columns_map.values(),
            index=df_old.columns,
            columns=['metric_val', 'metric_pval'] if isinstance(self.metric, TwoColumnTest) else ['metric_val']
        )

        result.name = self.metric.name
        return result


class CorrMatrix:
    """Computes the correlation between each pair of columns in the given dataframe
    and returns the result in a dataframe"""

    def __init__(self, metric: Union[TwoColumnMetric, TwoColumnTest]):
        self.metric = metric
        self.name = f'{metric.name}_matrix'

    def __call__(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        columns = df.columns
        matrix = pd.DataFrame(index=columns, columns=columns)

        for col_a, col_b in permutations(columns, 2):
            matrix[col_a][col_b] = self.metric(df[col_a], df[col_b])

        pval_matrix = None
        if isinstance(self.metric, TwoColumnTest):
            value_matrix = pd.DataFrame(index=columns, columns=columns)
            pval_matrix = pd.DataFrame(index=columns, columns=columns)

            for col_a, col_b in permutations(columns, 2):
                value_matrix[col_a][col_b], pval_matrix[col_a][col_b] = matrix[col_a][col_b]
        else:
            value_matrix = matrix

        return pd.DataFrame(value_matrix.astype(np.float32)), pval_matrix  # explicit casting for mypy


class DiffCorrMatrix:
    """Computes the correlation matrix for each of the given dataframes and return the difference
    between these matrices"""

    def __init__(self, metric: Union[TwoColumnMetric, TwoColumnTest]):
        self.corr_matrix = CorrMatrix(metric)
        self.name = f'diff_{metric.name}'

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[pd.DataFrame, None]:
        corr_matrix_old = self.corr_matrix(df=df_old)[0]
        corr_matrix_new = self.corr_matrix(df=df_new)[0]

        if corr_matrix_old is None or corr_matrix_new is None:
            return None

        return corr_matrix_old - corr_matrix_new
