"""This module contains the base classes for the metrics used across synthesized."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Type, Union

import pandas as pd

from ..check import Check, ColumnCheck


class _Metric(ABC):
    """
    An abstract base class from which more detailed metrics are derived.
    """
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name}

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.name}"


class OneColumnMetric(_Metric):
    """
    An abstract base class from which more specific single column metrics are derived.
    One column metric is a quantitative way of evaluating the data inside a single column.

    Example for a single column metric: mean.

    Usage:
        Create and configure the metric.
        >>> metric = OneColumnMetricChild(...)

        Evaluate on a single dataframe column.
        >>> metric(df["col_A"])
        0.5
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
    An abstract base class from which more specific two columns metrics are derived.
    Two column metric is a quantitative way of evaluating the data inside two separate columns.

    Example for two column metric: Euclidean Distance.

    Usage:
        Create and configure the metric.
        >>> metric = TwoColumnMetricChild(...)

        Evaluate on two dataframe columns.
        >>> metric(df['col_A'], df['col_B'])
        5

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
    An abstract base class from which more specific dataframe metrics are derived.
    A dataframe metric is a quantitative way of evaluating the data inside a single dataframe.

    Example for a dataframe metric: Ideal number of clusters.

    Usage:
        Create and configure the metric.
        >>> metric = DataFrameMetricChild(...)

        Evaluate on a dataframe.
        >>> metric(df)
        3
    """

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> Union[int, float, None]:
        pass


class TwoDataFrameMetric(_Metric):
    """
    An abstract base class from which more specific two-dataframe metrics are derived.
    A two-dataframe metric is a quantitative way of evaluating the data inside two separate dataframes.

    Example for a two-dataframe metric: predictive model accuracy.

    Usage:
        Create and configure the metric.
        >>> metric = TwoDataFrameMetricChild(...)

        Evalueate on two dataframes.
        >>> metric(df)
        0.90

    """

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> Union[int, float, None]:
        pass


class MetricFactory:
    """

    """
    # Concerns:
    # May not be the best module for it to sit.
    # Should it really be accessing a private class?

    _registry: Dict[str, Type[_Metric]] = {}

    @classmethod
    def update_registry(cls):
        pass

    @classmethod
    def metric_from_dict(cls):
        pass
