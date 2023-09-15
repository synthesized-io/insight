"""This module contains the base classes for the metrics used across synthesized."""
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

import pandas as pd

try:
    from sqlalchemy.orm import Session

    import insight.database.schema as model
    import insight.database.utils as utils
except ModuleNotFoundError:
    model = None  # type: ignore
    utils = None  # type: ignore
    Session = None  # type: ignore


from ..check import Check, ColumnCheck


class _Metric(ABC):
    """
    An abstract base class from which more detailed metrics are derived.
    """
    name: Optional[str] = None
    _registry: Dict[str, Type] = {}
    _session: Optional[Session] = utils.get_session() if utils is not None else None

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

    def _add_to_database(self,
                         value,
                         dataset_name: str,
                         dataset_rows: int = None,
                         dataset_cols: int = None,
                         category: str = None):
        """
        Adds the metric result to the database. The metric result should be specified as value.
        Args:
            value: The result of the metric
            dataset_name: The name of the dataset on which the metric was run.
            dataset_rows: Number of rows in the dataset.
            dataset_cols: Number of column in the dataset.
            category: The category of the metric.

        `version` and `run_id` are taken from `VERSION` and `RUN_ID` envvars.
        """
        version = os.getenv("VERSION")
        version = version if version else "Unversioned"
        run_id = os.getenv("RUN_ID")

        if model is None or utils is None:
            raise ModuleNotFoundError("The database module is not available. Please install it using the command: pip install 'insight[db]'")

        if self._session is None:
            raise RuntimeError("Called a database function when no database exists.")

        if self.name is None:
            raise AttributeError("Every initializeable subclass of _Metric must have a name string")

        if hasattr(value, 'item'):
            value = value.item()

        with self._session as session:
            metric_id = utils.get_metric_id(self.name, session, category=category)
            version_id = utils.get_version_id(version, session)
            dataset_id = utils.get_df_id(dataset_name, session, num_rows=dataset_rows, num_columns=dataset_cols)
            result = model.Result(
                metric_id=metric_id, dataset_id=dataset_id, version_id=version_id, value=value, run_id=run_id
            )
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

    def __init__(self, check: Check = ColumnCheck(), upload_to_database=True):
        self.check = check
        self._upload_to_database = upload_to_database and self._session is not None

    def _disable_database_upload(self):
        """
        Disables logging metrics into the database until enable_database_upload_if_exists is called. The call will have
        no effect if database integration was already disabled.
        """
        self._upload_to_database = False

    def _enable_database_upload_if_exists(self):
        """
        Enables logging metrics into the database until the next call. The call will have no effect if database
        integration was already disabled.
        """
        self._upload_to_database = self._session is not None

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
                 dataset_name: str = None):
        if not self.check_column_types(sr, self.check):
            value = None
        else:
            value = self._compute_metric(sr)

        if self._upload_to_database:
            dataset_name = "Series_" + str(sr.name) if dataset_name is None else dataset_name
            self._add_to_database(value,
                                  dataset_name,
                                  dataset_rows=len(sr),
                                  category='OneColumnMetric',
                                  dataset_cols=1)

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

    def __init__(self, check: Check = ColumnCheck(), upload_to_database=True):
        self.check = check
        self._upload_to_database = upload_to_database and self._session is not None

    def _disable_database_upload(self):
        """
        Disables logging metrics into the database until enable_database_upload_if_exists is called. The call will have
        no effect if database integration was already disabled.
        """
        self._upload_to_database = False

    def _enable_database_upload_if_exists(self):
        """
        Enables logging metrics into the database until the next call. The call will have no effect if database
        integration was already disabled.
        """
        self._upload_to_database = self._session is not None

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

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, dataset_name: str = None):
        if not self.check_column_types(sr_a, sr_b, self.check):
            value = None
        else:
            value = self._compute_metric(sr_a, sr_b)

        if self._upload_to_database:
            dataset_name = "Series_" + str(sr_a.name) if dataset_name is None else dataset_name
            self._add_to_database(value,
                                  dataset_name,
                                  dataset_rows=len(sr_a),
                                  category='TwoColumnMetric',
                                  dataset_cols=1)

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

    def __call__(self, df: pd.DataFrame, dataset_name: str = None) -> Union[pd.DataFrame, None]:
        result = self._compute_result(df)
        dataset_rows = df.shape[0]
        dataset_cols = df.shape[1]
        if self._session is not None:
            if dataset_name is None:
                dataset_name = df.attrs.get("name")  # Explicit cast for mypy.
                if dataset_name is None:
                    raise AttributeError(
                        "Must specify the name of the dataset name as a parameter to upload to database.")

            self._add_to_database(self.summarize_result(result),
                                  dataset_name,
                                  dataset_rows=dataset_rows,
                                  dataset_cols=dataset_cols,
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
                 dataset_name: str = None) -> Union[pd.DataFrame, None]:
        result = self._compute_result(df_old, df_new)
        dataset_rows = df_old.shape[0]
        dataset_cols = df_old.shape[1]
        if self._session is not None:
            if dataset_name is None:
                dataset_name = df_old.attrs.get("name")  # Explicit cast for mypy.
                if dataset_name is None:
                    raise AttributeError(
                        "Must specify the name of the dataset name as a parameter to upload to database.")

            self._add_to_database(self.summarize_result(result),
                                  dataset_name,
                                  dataset_cols=dataset_cols,
                                  dataset_rows=dataset_rows,
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
