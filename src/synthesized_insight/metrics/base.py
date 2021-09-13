from abc import ABC, abstractclassmethod, abstractmethod
from itertools import permutations
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..check import Check, ColumnCheck


class _Metric(ABC):
    name: Optional[str] = None
    tags: Sequence[str] = []

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.name}"


class OneColumnMetric(_Metric):

    def __init__(self, check: Check = ColumnCheck()):
        self.check = check

    @abstractclassmethod
    def check_column_types(cls, sr: pd.Series, check: Check = ColumnCheck()) -> bool:
        ...

    @abstractmethod
    def _compute_metric(self, sr: pd.Series):
        ...

    def __call__(self, sr: pd.Series):
        if not self.check_column_types(sr, self.check):
            return None
        return self._compute_metric(sr)


class TwoColumnMetric(_Metric):
    def __init__(self, check: Check = ColumnCheck()):
        self.check = check

    @abstractclassmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()):
        ...

    @abstractmethod
    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        ...

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series):
        if not self.check_column_types(sr_a, sr_b, self.check):
            return None
        return self._compute_metric(sr_a, sr_b)


class TwoColumnTest(_Metric):
    def __init__(self,
                 check: Check = ColumnCheck(),
                 alternative: str = 'two-sided'):

        if alternative not in ('two-sided', 'greater', 'less'):
            raise ValueError("'alternative' argument must be one of 'two-sided', 'greater', 'less'")

        self.check = check
        self.alternative = alternative

    @abstractclassmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()) -> bool:
        ...

    @abstractmethod
    def _compute_test(self,
                      sr_a: pd.Series,
                      sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        ...

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        if not self.check_column_types(sr_a, sr_b, self.check):
            return None, None

        return self._compute_test(sr_a, sr_b)


class TwoColumnMap(_Metric):
    """Compares columns with the same name from two given dataframes and return a DataFrame
    with index as the column name and the columns as metric_value and metric_pval(if applicable)"""

    def __init__(self, metric: Union[TwoColumnMetric, TwoColumnTest]):
        self.metric = metric
        self.name = f'{metric.name}_map'
        super().__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[pd.DataFrame, None]:
        if df_old is None or df_new is None:
            return None

        columns_map = {col: self.metric(df_old[col], df_new[col]) for col in df_old.columns}

        result = pd.DataFrame(
            data=columns_map.values(),
            index=df_old.columns,
            columns=['metric_val', 'metric_pval'] if isinstance(self.metric, TwoColumnTest) else ['metric_val']
        )

        return result


class CorrMatrix(_Metric):
    """Computes the correlation between each pair of columns in the given dataframe
    and returns the result in a dataframe"""

    def __init__(self, metric: Union[TwoColumnMetric, TwoColumnTest]):
        self.metric = metric
        self.name = f'{metric.name}_matrix'
        super().__init__()

    def __call__(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
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


class DiffCorrMatrix(_Metric):
    """Computes the correlation matrix for each of the given dataframes and return the difference
    between these matrices"""

    def __init__(self, metric: Union[TwoColumnMetric, TwoColumnTest]):
        self.corr_metric = CorrMatrix(metric)
        self.name = f'diff_{metric.name}'
        super().__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[pd.DataFrame, None]:
        if df_old is None or df_new is None:
            return None

        corr_matrix_old = self.corr_metric(df=df_old)[0]
        corr_matrix_new = self.corr_metric(df=df_new)[0]

        if corr_matrix_old is None or corr_matrix_new is None:
            return None

        return corr_matrix_old - corr_matrix_new


class ModellingMetric(_Metric):

    @abstractmethod
    def __call__(self,
                 y_true: np.ndarray,
                 y_pred: Optional[np.ndarray] = None) -> Union[float, None]:
        pass


class ClassificationMetric(ModellingMetric):
    tags = ["modelling", "classification"]
    plot = False

    def __init__(self, multiclass: bool = False):
        self.multiclass = multiclass

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> float:
        raise NotImplementedError


class ClassificationPlotMetric(ModellingMetric):
    tags = ["modelling", "classification", "plot"]
    plot = True

    def __init__(self, multiclass: bool = False):
        self.multiclass = multiclass

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None) -> Any:
        raise NotImplementedError


class RegressionMetric(ModellingMetric):
    tags = ["modelling", "regression"]

    def __init__(self):
        # Contains nothing atm but matches other two Modelling metrics.
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> float:
        raise NotImplementedError


class DataFrameMetric(_Metric):

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> Union[int, float, None]:
        pass


class TwoDataFrameMetric(_Metric):

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[int, float, None]:
        pass
