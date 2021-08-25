from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..check import ColumnCheck


class _Metric(ABC):
    name: Optional[str] = None
    tags: Sequence[str] = []

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.name}"


class OneColumnMetric(_Metric):

    def __init__(self, check: ColumnCheck = None):
        if check is None:
            self.check = ColumnCheck()
        else:
            self.check = check

    @abstractclassmethod
    def check_column_types(cls, check: ColumnCheck, sr: pd.Series) -> bool:
        ...

    @abstractmethod
    def _compute_metric(self, sr: pd.Series):
        ...

    def __call__(self, sr: pd.Series):
        if not self.check_column_types(self.check, sr):
            return None
        return self._compute_metric(sr)


class TwoColumnMetric(_Metric):
    def __init__(self, check: ColumnCheck = None):
        if check is None:
            self.check = ColumnCheck()
        else:
            self.check = check

    @abstractclassmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        ...

    @abstractmethod
    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        ...

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series):
        if not self.check_column_types(self.check, sr_a, sr_b):
            return None
        return self._compute_metric(sr_a, sr_b)


class TwoColumnMetricTest(_Metric):
    def __init__(self, check: ColumnCheck = None, metric_cls_obj: Union[TwoColumnMetric, None] = None):
        if check is None:
            self.check = ColumnCheck()
        else:
            self.check = check
        self.metric_cls_obj = metric_cls_obj
        self.p_value: Union[int, float, None] = None

    @abstractclassmethod
    def check_column_types(cls, check: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        ...

    @abstractmethod
    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series) -> Union[int, float, None]:
        ...

    def _compute_p_value(self,
                         sr_a: pd.Series,
                         sr_b: pd.Series,
                         metric_value: float,
                         alternative: str = 'two-sided') -> Union[int, float, None]:
        return self.p_value

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series) -> Tuple[Union[int, float, None], Union[int, float, None]]:
        if not self.check_column_types(self.check, sr_a, sr_b):
            return None, None

        metric_value = self._compute_metric(sr_a, sr_b)
        if metric_value is None:
            return None, None

        p_value = self._compute_p_value(sr_a, sr_b, metric_value)
        return metric_value, p_value


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
