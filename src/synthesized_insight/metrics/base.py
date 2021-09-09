from abc import ABC, abstractclassmethod, abstractmethod
from itertools import combinations, permutations
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


class ColumnComparisonVector(_Metric):
    """Compares columns with the same name from two given dataframes and return a series
    with index as the column name and the row value as the comparison metric value"""

    def __init__(self, metric: Union[TwoColumnMetric, TwoColumnTest], return_p_val: bool = False):
        self.metric = metric
        self.name = f'{metric.name}_vector'
        self.return_p_val = False
        super().__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[pd.Series, None]:
        if df_old is None or df_new is None:
            return None

        result = pd.Series(
            data=[self.metric(df_old[col], df_new[col]) for col in df_old.columns],
            index=df_old.columns,
            name=self.metric.name
        )

        if isinstance(self.metric, TwoColumnTest) and not self.return_p_val:
            result = pd.Series(result.apply(lambda x: x[0]))  # fetch only the metric value, not p-value

        return result


class TwoColumnMetricMatrix(_Metric):
    """Computes the correlation between each pair of columns in the given dataframe
    and returns the result in a dataframe"""

    def __init__(self, metric: Union[TwoColumnMetric, TwoColumnTest]):
        self.metric = metric
        self.name = f'{metric.name}_matrix'
        super(TwoColumnMetricMatrix, self).__init__()

    def _get_metric_value(self, sr_a: pd.Series, sr_b: pd.Series):
        res = self.metric(sr_a, sr_b)
        if isinstance(self.metric, TwoColumnTest):
            return res[0]
        return res

    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        columns = df.columns
        matrix = pd.DataFrame(index=columns, columns=columns)

        if 'symmetric' in self.tags:
            for col_a, col_b in combinations(columns, 2):
                matrix[col_a][col_b] = matrix[col_b][col_a] = self._get_metric_value(df[col_a], df[col_b])
        else:
            for col_a, col_b in permutations(columns, 2):
                matrix[col_a][col_b] = self._get_metric_value(df[col_a], df[col_b])

        return pd.DataFrame(matrix.astype(np.float32))  # need explicit casting for mypy


class DiffMetricMatrix(_Metric):
    """Computes the correlation matrix for each of the given dataframes and return the difference
    between these matrices"""

    def __init__(self, metric: TwoColumnMetricMatrix):
        self.metric = metric
        self.name = f'diff_{metric.name}'
        super().__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[pd.DataFrame, None]:
        if df_old is None or df_new is None:
            return None

        matrix_old = self.metric(df=df_old)
        matrix_new = self.metric(df=df_new)

        if matrix_old is None or matrix_new is None:
            return None

        return matrix_new - matrix_old


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
