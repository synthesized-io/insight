from typing import Optional, List
from abc import ABC, abstractclassmethod, abstractmethod

import pandas as pd

from src.synthesized_insight import ColumnCheck


class _Metric(ABC):
    name: Optional[str] = None
    tags: List[str] = []

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.name}"


class OneColumnMetric(_Metric):

    def __init__(self, checker: ColumnCheck = None):
        if checker is None:
            self.checker = ColumnCheck()
        else:
            self.checker = checker

    @abstractclassmethod
    def check_column_types(cls, checker: ColumnCheck, sr: pd.Series) -> bool:
        ...

    @abstractmethod
    def _compute_metric(self, sr: pd.Series):
        ...

    def __call__(self, sr: pd.Series):
        if not self.check_column_types(self.checker, sr):
            return None
        return self._compute_metric(sr)


class TwoColumnMetric(_Metric):
    def __init__(self, checker: ColumnCheck = None):
        if checker is None:
            self.checker = ColumnCheck()
        else:
            self.checker = checker

    @abstractclassmethod
    def check_column_types(cls, checker: ColumnCheck, sr_a: pd.Series, sr_b: pd.Series):
        ...

    @abstractmethod
    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        ...

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series):
        if not self.check_column_types(self.checker, sr_a, sr_b):
            return None
        return self._compute_metric(sr_a, sr_b)
