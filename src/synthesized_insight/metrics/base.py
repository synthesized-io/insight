from abc import ABC, abstractclassmethod, abstractmethod
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
