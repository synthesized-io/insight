"""This module contains the base classes for the metrics used across synthesized."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

import pandas as pd
from sqlalchemy.orm import Session

import insight.database.schema as model
import insight.database.utils as utils

from ..check import Check, ColumnCheck

VERSION = "v1.10"


class _Metric(ABC):
    """
    An abstract base class from which more detailed metrics are derived.
    """
    name: Optional[str] = None
    _registry: Dict[str, Type] = {}

    def __init_subclass__(cls):
        if cls.name is not None and cls.name not in _Metric._registry:
            _Metric._registry.update({cls.name: cls})

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the metric into a dictionary representation of itself.
        Returns: a dictionary with key value pairs that represent the metric.
        """
        return {'name': self.name}

    @classmethod
    def from_dict(cls, bluprnt: Dict[str, Any], check: Check = None):
        """
        Given a dictionary, builds and returns a metric that corresponds to the specified metric with the given metric
        parameters.

        Expected dictionary format:
        {'name': 'my_metric',
         'metric_param1': param1,
         'metric_param2': param2
         ...}
        """
        bluprnt_params = {key: val for key, val in bluprnt.items() if key != 'name'}
        if check is not None:
            bluprnt.update({'check': check})

        metric = _Metric._registry[bluprnt['name']](**bluprnt_params)
        return metric

    def add_to_database(self,
                        value,
                        dataset_name: str,
                        session: Session,
                        dataset_rows: int = None,
                        dataset_col: int = None,
                        category: str = None,
                        version: str = "v1.10"):
        """

        Args:
            value:
            dataset_name:
            session:
            dataset_rows:
            dataset_col:
            category:
            version:
        """
        metric_id = utils.get_metric_id(self, session, category=category)
        version_id = utils.get_version_id(version, session)
        dataset_id = utils.get_df_id(dataset_name, session, num_rows=dataset_rows, num_columns=dataset_col)

        with session:
            result = model.Result(metric_id=metric_id, dataset_id=dataset_id, version_id=version_id, value=value)
            session.add(result)
            session.commit()


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
        """
        Check that the column type of the series satisfy the requirements of the metric.
        Args:
            sr: the series to check.
            check: the check that is used.
        """
        ...

    @abstractmethod
    def _compute_metric(self, sr: pd.Series):
        ...

    def __call__(self,
                 sr: pd.Series,
                 session: Session = None,
                 dataset_name: str = None,
                 num_rows: int = None,
                 version: str = VERSION):
        if not self.check_column_types(sr, self.check):
            value = None
        else:
            value = self._compute_metric(sr)

        if session is not None:
            dataset_name = "Series_" + str(sr.name) if dataset_name is None else dataset_name
            self.add_to_database(value,
                                 dataset_name,
                                 session,
                                 version=version,
                                 dataset_rows=num_rows,
                                 category='OneColumnMetric',
                                 dataset_col=1)

        return value


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
        """
        Check that the column types of the serieses satisfy the requirements of the metric.
        Args:
            sr_a: the first series to check.
            sr_b: the second series to check.
            check: the check that is used.
        """
        ...

    @abstractmethod
    def _compute_metric(self, sr_a: pd.Series, sr_b: pd.Series):
        ...

    def __call__(self,
                 sr_a: pd.Series,
                 sr_b: pd.Series,
                 session: Session = None,
                 dataset_name: str = None,
                 num_rows: int = None,
                 version: str = VERSION):
        if not self.check_column_types(sr_a, sr_b, self.check):
            value = None
        else:
            value = self._compute_metric(sr_a, sr_b)

        if session is not None:
            dataset_name = "Series_" + str(sr_a.name) if dataset_name is None else dataset_name
            self.add_to_database(value,
                                 dataset_name,
                                 session,
                                 version=version,
                                 dataset_rows=num_rows,
                                 category='TwoColumnMetric',
                                 dataset_col=1)

        return value


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
    def __call__(self,
                 df: pd.DataFrame,
                 dataset_name: str = None,
                 session: Session = None,
                 version: str = VERSION) -> Union[pd.DataFrame, None]:
        result = self._compute_result(df)
        if session is not None:
            if dataset_name is None:
                try:
                    dataset_name = df.name
                except AttributeError:
                    raise AttributeError(
                        "Must specify the name of the dataset either as a DataFrame.name or as a parameter.")

            self.add_to_database(self.summarize_result(result),
                                 dataset_name,
                                 session,
                                 version=version,
                                 category='DataFrameMetric')
        return result

    @abstractmethod
    def _compute_result(self, df: pd.DataFrame):
        ...

    @abstractmethod
    def summarize_result(self, result):
        """
        Give a single value that summarizes the result of the metric.
        Args:
            result: the result of the metric computation.
        """
        ...


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
    def __call__(self,
                 df_old: pd.DataFrame,
                 df_new: pd.DataFrame,
                 dataset_name: str = None,
                 session: Session = None,
                 version: str = VERSION) -> Union[pd.DataFrame, None]:
        result = self._compute_result(df_old, df_new)
        if session is not None:
            if dataset_name is None:
                try:
                    dataset_name = str(df_old.name)     # Explicit cast for mypy.
                except AttributeError:
                    try:
                        dataset_name = str(df_new.name)     # Explicit cast for mypy.
                    except AttributeError:
                        raise AttributeError(
                            "Must specify the name of the dataset either as a DataFrame.name or as a parameter.")

            self.add_to_database(self.summarize_result(result),
                                 dataset_name,
                                 session,
                                 version=version,
                                 category='TwoDataFrameMetrics')
        return result

    @abstractmethod
    def _compute_result(self, df_old, df_new):
        ...

    @abstractmethod
    def summarize_result(self, result):
        """
        Give a single value that summarizes the result of the metric.
        Args:
            result: the result of the metric computation.
        """
        ...
