from typing import Optional, Sequence, Union
from abc import ABC, abstractclassmethod, abstractmethod

import pandas as pd

from src.synthesized_insight import ColumnCheck


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
