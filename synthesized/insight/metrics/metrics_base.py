"""This module contains metrics with different 'levels' of detail."""
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Union
from itertools import combinations

import pandas as pd


def _register(metric, registry):
    if metric.name is None:
        raise ValueError("Metric 'name' not specified.")
    registry[metric.name] = metric


class _Metric(ABC):
    ALL: Dict[str,  Type['_Metric']] = {}
    name: Union[str, None] = None
    tags: List[str] = []

    def __init__(self):
        _register(self, _Metric.ALL)


class ColumnMetric(_Metric):
    ALL: Dict[str, Type['ColumnMetric']] = {}

    def __init__(self):
        super(ColumnMetric, self).__init__()
        _register(self, ColumnMetric.ALL)

        AllColumnMetricAdapter(self, 'min')
        AllColumnMetricAdapter(self, 'max')
        AllColumnMetricAdapter(self, 'avg')
        DiffColumnMetricAdapter(self)

    @abstractmethod
    def __call__(self, df: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        pass


class TwoColumnMetric(_Metric):
    ALL: Dict[str, Type['TwoColumnMetric']] = {}

    def __init__(self):
        super(_Metric, self).__init__()
        _register(self, TwoColumnMetric.ALL)

        AllTwoColumnMetricAdapter(self, 'min')
        AllTwoColumnMetricAdapter(self, 'max')
        AllTwoColumnMetricAdapter(self, 'avg')
        DiffTwoColumnMetricAdapter(self)

    @abstractmethod
    def __call__(self, df: pd.DataFrame, col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        pass


class DataFrameMetric(_Metric):
    ALL: Dict[str, Type['DataFrameMetric']] = {}

    def __init__(self):
        super(_Metric, self).__init__()
        _register(self, DataFrameMetric.ALL)

        DiffDataFrameMetricAdapter(self)

    @abstractmethod
    def __call__(self, df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        pass


class ColumnComparison(_Metric):
    ALL: Dict[str, Type['ColumnComparison']] = {}

    def __init__(self):
        super(_Metric, self).__init__()
        _register(self, ColumnComparison.ALL)

        AllColumnComparisonAdapter(self, 'min')
        AllColumnComparisonAdapter(self, 'max')
        AllColumnComparisonAdapter(self, 'avg')

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        pass


class TwoColumnComparison(_Metric):
    ALL: Dict[str, Type['TwoColumnComparison']] = {}

    def __init__(self):
        super(_Metric, self).__init__()
        _register(self, TwoColumnComparison.ALL)

        AllTwoColumnComparisonAdapter(self, 'min')
        AllTwoColumnComparisonAdapter(self, 'max')
        AllTwoColumnComparisonAdapter(self, 'avg')

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame,
                 col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        pass


class DataFrameComparison(_Metric):
    ALL: Dict[str, Type['DataFrameComparison']] = {}

    def __init__(self):
        super(_Metric, self).__init__()
        _register(self, DataFrameComparison.ALL)

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        pass


# Adapters to transform Metrics into Composite Metrics
# ----------------------------------------------------

class AllColumnMetricAdapter(DataFrameMetric):
    def __init__(self, metric: ColumnMetric, summary_type: str):
        if summary_type not in ['min', 'max', 'avg']:
            raise ValueError
        self.summary_type = summary_type
        self.metric = metric
        self.name = f'{summary_type}_{self.name}'
        super(DataFrameMetric, self).__init__()

    def __call__(self, df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        values: List[Union[int, float]] = []
        for col in df.columns:
            value = self.metric(df, col, **kwargs)
            if value is not None:
                values.append(value)

        if self.summary_type == 'min':
            return min(values) if len(values) > 0 else None
        elif self.summary_type == 'max':
            return max(values) if len(values) > 0 else None
        else:
            assert self.summary_type == 'avg'
            return sum(values) / len(values) if len(values) > 0 else None


class DiffColumnMetricAdapter(ColumnComparison):
    def __init__(self, metric: ColumnMetric):
        self.metric = metric
        self.name = f'diff_{self.name}'
        super(ColumnComparison, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        value_old = self.metric(df_old, col_name, **kwargs)
        value_new = self.metric(df_new, col_name, **kwargs)
        if value_old is None or value_new is None:
            return None
        else:
            return value_new - value_old


class AllTwoColumnMetricAdapter(DataFrameMetric):
    def __init__(self, metric: TwoColumnMetric, summary_type: str):
        if summary_type not in ['min', 'max', 'avg']:
            raise ValueError
        self.summary_type = summary_type
        self.metric = metric
        self.name = f'{summary_type}_{self.name}'
        super(DataFrameMetric, self).__init__()

    def __call__(self, df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        values: List[Union[int, float]] = []
        for col_a, col_b in combinations(df.columns, 2):
            value = self.metric(df, col_a, col_b, **kwargs)
            if value is not None:
                values.append(value)

        if self.summary_type == 'min':
            return min(values) if len(values) > 0 else None
        elif self.summary_type == 'max':
            return max(values) if len(values) > 0 else None
        else:
            assert self.summary_type == 'avg'
            return sum(values) / len(values) if len(values) > 0 else None


class DiffTwoColumnMetricAdapter(TwoColumnComparison):
    def __init__(self, metric: TwoColumnMetric):
        self.metric = metric
        self.name = f'diff_{self.name}'
        super(TwoColumnComparison, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame,
                 col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        value_old = self.metric(df_old, col_a_name, col_b_name, **kwargs)
        value_new = self.metric(df_new, col_a_name, col_b_name, **kwargs)
        if value_old is None or value_new is None:
            return None
        else:
            return value_new - value_old


class DiffDataFrameMetricAdapter(DataFrameComparison):
    def __init__(self, metric: DataFrameMetric):
        self.metric = metric
        self.name = f'diff_{self.name}'
        super(DataFrameComparison, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        value_old = self.metric(df_old, **kwargs)
        value_new = self.metric(df_new, **kwargs)
        if value_old is None or value_new is None:
            return None
        else:
            return value_new - value_old


class AllColumnComparisonAdapter(DataFrameComparison):
    def __init__(self, metric: ColumnComparison, summary_type: str):
        if summary_type not in ['min', 'max', 'avg']:
            raise ValueError
        self.summary_type = summary_type
        self.metric = metric
        self.name = f'{summary_type}_{self.name}'
        super(DataFrameComparison, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        values: List[Union[int, float]] = []
        for col in df_old.columns:
            value = self.metric(df_old, df_new, col)
            if value is not None:
                values.append(value)
        if self.summary_type == 'min':
            return min(values) if len(values) > 0 else None
        elif self.summary_type == 'max':
            return max(values) if len(values) > 0 else None
        else:
            assert self.summary_type == 'avg'
            return sum(values) / len(values) if len(values) > 0 else None


class AllTwoColumnComparisonAdapter(DataFrameComparison):
    def __init__(self, metric: TwoColumnComparison, summary_type: str):
        if summary_type not in ['min', 'max', 'avg']:
            raise ValueError
        self.summary_type = summary_type
        self.metric = metric
        self.name = f'{summary_type}_{self.name}'
        super(DataFrameComparison, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        values: List[Union[int, float]] = []
        for col_a, col_b in combinations(df_old.columns, 2):
            value = self.metric(df_old, df_new, col_a, col_b)
            if value is not None:
                values.append(value)

        if self.summary_type == 'min':
            return min(values) if len(values) > 0 else None
        elif self.summary_type == 'max':
            return max(values) if len(values) > 0 else None
        else:
            assert self.summary_type == 'avg'
            return sum(values) / len(values) if len(values) > 0 else None
