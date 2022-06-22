from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import pandas as pd

from ..check import Check, ColumnCheck


class _Metric(ABC):
    # TODO: Improve the docstring.
    """
    An abstract base class from which more detailed metrics are derived.
    """
    name: Optional[str] = None
    tags: Sequence[str] = []

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.name}"


class OneColumnMetric(_Metric):
    """
    An abstract base class from which more specific single column metrics are derived.
    It is the respondibility of the  subclass to define check_column_types,
    and _compute_metric.
    """
    def __init__(self, check: Check = ColumnCheck()):
        self.check = check

    @classmethod
    @abstractmethod
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
    """

    """
    def __init__(self, check: Check = ColumnCheck()):
        self.check = check

    @classmethod
    @abstractmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()):
        ...

    @abstractmethod
    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        ...

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series):
        if not self.check_column_types(sr_a, sr_b, self.check):
            return None
        return self._compute_metric(sr_a, sr_b)


class DataFrameMetric(_Metric):
    """

    """

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> Union[int, float, None]:
        pass


class TwoDataFrameMetric(_Metric):
    """

    """

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[int, float, None]:
        pass
