"""This module contains metrics with different 'levels' of detail."""
from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Type, Union
from itertools import combinations, permutations

import pandas as pd
import numpy as np

from ..dataset import categorical_or_continuous_values


def _register(metric, cls):
    registry = cls.ALL
    # print(f'Registering metric: {metric.name} in {cls.__name__} registry. ')
    if metric.name is None:
        raise ValueError("Metric 'name' not specified.")
    # if metric.name in registry:
    #     raise ValueError("Metric 'name' already exists.")
    registry[metric.name] = metric


class _Metric(ABC):
    ALL: Mapping[str,  Type['_Metric']] = {}
    name: Union[str, None] = None
    tags: List[str] = []

    def __init__(self):
        _register(self, _Metric)
        super(_Metric, self).__init__()


class ColumnMetric(_Metric):
    ALL: Dict[str, Type['ColumnMetric']] = {}

    def __init__(self):
        _register(self, ColumnMetric)

        AggregateColumnMetricAdapter(self, 'min')
        AggregateColumnMetricAdapter(self, 'max')
        AggregateColumnMetricAdapter(self, 'avg')
        DiffColumnMetricAdapter(self)
        ColumnMetricVector(self)

        super(ColumnMetric, self).__init__()

    @abstractmethod
    def __call__(self, sr: pd.Series, **kwargs) -> Union[int, float, None]:
        pass


class TwoColumnMetric(_Metric):
    ALL: Dict[str, Type['TwoColumnMetric']] = {}

    def __init__(self):
        _register(self, TwoColumnMetric)

        AggregateTwoColumnMetricAdapter(self, 'min')
        AggregateTwoColumnMetricAdapter(self, 'max')
        AggregateTwoColumnMetricAdapter(self, 'avg')
        DiffTwoColumnMetricAdapter(self)
        TwoColumnMetricMatrix(self)

        super(TwoColumnMetric, self).__init__()

    def check_column_types(self, sr_a: pd.Series, sr_b: pd.Series, **kwargs):
        df = pd.DataFrame([sr_a, sr_b])
        if "nominal" in self.tags:
            categorical, _ = categorical_or_continuous_values(kwargs.get('vf', df))
            categorical_columns = [v.name for v in categorical]

            if sr_a.name not in categorical_columns or sr_b.name not in categorical_columns:
                return False

        if "ordinal" in self.tags:
            _, continuous = categorical_or_continuous_values(kwargs.get('vf', df))
            continuous_columns = [v.name for v in continuous]

            if sr_a.name not in continuous_columns or sr_b.name not in continuous_columns:
                return False

        if kwargs.get('continuous_input_only', False):
            _, continuous = categorical_or_continuous_values(kwargs.get('vf', df))
            continuous_columns = [v.name for v in continuous]

            if sr_a.name not in continuous_columns:
                return False

        if kwargs.get('categorical_output_only', False):
            categorical, _ = categorical_or_continuous_values(kwargs.get('vf', df))
            categorical_columns = [v.name for v in categorical]

            if sr_b.name not in categorical_columns:
                return False

        return True

    @abstractmethod
    def __call__(self, sr_a: pd.Series, sr_b: pd.Series, **kwargs) -> Union[int, float, None]:
        pass


class DataFrameMetric(_Metric):
    ALL: Dict[str, Type['DataFrameMetric']] = {}

    def __init__(self):
        _register(self, DataFrameMetric)

        DiffDataFrameMetricAdapter(self)

        super(DataFrameMetric, self).__init__()

    @abstractmethod
    def __call__(self, df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        pass


class ColumnComparison(_Metric):
    ALL: Dict[str, Type['ColumnComparison']] = {}

    def __init__(self):
        _register(self, ColumnComparison)

        AggregateColumnComparisonAdapter(self, 'min')
        AggregateColumnComparisonAdapter(self, 'max')
        AggregateColumnComparisonAdapter(self, 'avg')
        ColumnComparisonVector(self)

        super(ColumnComparison, self).__init__()

    def check_column_types(self, df_old, df_new, col_name, **kwargs):
        if "nominal" in self.tags:
            categorical, _ = categorical_or_continuous_values(kwargs.get('vf', df_old))
            categorical_columns = [v.name for v in categorical]

            if col_name not in categorical_columns:
                return False

        if "ordinal" in self.tags:
            _, continuous = categorical_or_continuous_values(kwargs.get('vf', df_old))
            continuous_columns = [v.name for v in continuous]

            if col_name not in continuous_columns:
                return False

        return True

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        pass


class TwoColumnComparison(_Metric):
    ALL: Dict[str, Type['TwoColumnComparison']] = {}

    def __init__(self):
        _register(self, TwoColumnComparison)

        AggregateTwoColumnComparisonAdapter(self, 'min')
        AggregateTwoColumnComparisonAdapter(self, 'max')
        AggregateTwoColumnComparisonAdapter(self, 'avg')
        TwoColumnComparisonMatrix(self)

        super(TwoColumnComparison, self).__init__()

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame,
                 col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        pass


class DataFrameComparison(_Metric):
    ALL: Dict[str, Type['DataFrameComparison']] = {}

    def __init__(self):
        _register(self, DataFrameComparison)

        super(DataFrameComparison, self).__init__()

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        pass


# Adapters to transform Metrics into Composite Metrics
# ----------------------------------------------------

class AggregateColumnMetricAdapter(DataFrameMetric):
    def __init__(self, metric: ColumnMetric, summary_type: str):
        if summary_type not in ['min', 'max', 'avg']:
            raise ValueError
        self.summary_type = summary_type
        self.metric = metric
        self.name = f'{summary_type}_{metric.name}'
        super(AggregateColumnMetricAdapter, self).__init__()

    def __call__(self, df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        values: List[Union[int, float]] = []
        for col in df.columns:
            value = self.metric(df[col], **kwargs)
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
        self.name = f'diff_{metric.name}'
        super().__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        value_old = self.metric(df_old[col_name], **kwargs)
        value_new = self.metric(df_new[col_name], **kwargs)
        if value_old is None or value_new is None:
            return None
        else:
            return value_new - value_old


class AggregateTwoColumnMetricAdapter(DataFrameMetric):
    def __init__(self, metric: TwoColumnMetric, summary_type: str):
        if summary_type not in ['min', 'max', 'avg']:
            raise ValueError
        self.summary_type = summary_type
        self.metric = metric
        self.name = f'{summary_type}_{metric.name}'
        super(AggregateTwoColumnMetricAdapter, self).__init__()

    def __call__(self, df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        values: List[Union[int, float]] = []
        for col_a, col_b in combinations(df.columns, 2):
            value = self.metric(df[col_a], df[col_b], **kwargs)
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
        self.name = f'diff_{metric.name}'
        super(DiffTwoColumnMetricAdapter, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame,
                 col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        value_old = self.metric(df_old[col_a_name], df_old[col_b_name], **kwargs)
        value_new = self.metric(df_new[col_a_name], df_new[col_b_name], **kwargs)
        if value_old is None or value_new is None:
            return None
        else:
            return value_new - value_old


class DiffDataFrameMetricAdapter(DataFrameComparison):
    def __init__(self, metric: DataFrameMetric):
        self.metric = metric
        self.name = f'diff_{metric.name}'
        super(DiffDataFrameMetricAdapter, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        value_old = self.metric(df_old, **kwargs)
        value_new = self.metric(df_new, **kwargs)
        if value_old is None or value_new is None:
            return None
        else:
            return value_new - value_old


class AggregateColumnComparisonAdapter(DataFrameComparison):
    def __init__(self, metric: ColumnComparison, summary_type: str):
        if summary_type not in ['min', 'max', 'avg']:
            raise ValueError
        self.summary_type = summary_type
        self.metric = metric
        self.name = f'{summary_type}_{metric.name}'
        super(AggregateColumnComparisonAdapter, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        values: List[Union[int, float]] = []
        for col in df_old.columns:
            value = self.metric(df_old, df_new, col, **kwargs)
            if value is not None:
                values.append(value)
        if self.summary_type == 'min':
            return min(values) if len(values) > 0 else None
        elif self.summary_type == 'max':
            return max(values) if len(values) > 0 else None
        else:
            assert self.summary_type == 'avg'
            return sum(values) / len(values) if len(values) > 0 else None


class AggregateTwoColumnComparisonAdapter(DataFrameComparison):
    def __init__(self, metric: TwoColumnComparison, summary_type: str):
        if summary_type not in ['min', 'max', 'avg']:
            raise ValueError
        self.summary_type = summary_type
        self.metric = metric
        self.name = f'{summary_type}_{metric.name}'
        super(AggregateTwoColumnComparisonAdapter, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        values: List[Union[int, float]] = []
        for col_a, col_b in combinations(df_old.columns, 2):
            value = self.metric(df_old, df_new, col_a, col_b, **kwargs)
            if value is not None:
                values.append(value)

        if self.summary_type == 'min':
            return min(values) if len(values) > 0 else None
        elif self.summary_type == 'max':
            return max(values) if len(values) > 0 else None
        else:
            assert self.summary_type == 'avg'
            return sum(values) / len(values) if len(values) > 0 else None


# Vectors that return a series instead of one value
# ----------------------------------------------------
class _Vector(ABC):
    ALL: Mapping[str, Type['_Vector']] = {}
    name: Union[str, None] = None
    tags: List[str] = []

    def __init__(self):
        _register(self, _Vector)
        super(_Vector, self).__init__()


class ColumnVector(_Vector):
    ALL: Mapping[str, Type['ColumnVector']] = {}

    def __init__(self):
        _register(self, ColumnVector)
        super().__init__()

    @abstractmethod
    def __call__(self, sr: pd.Series, **kwargs) -> Union[pd.Series, None]:
        pass


class DataFrameVector(_Vector):
    ALL: Mapping[str, Type['DataFrameVector']] = {}

    def __init__(self):
        _register(self, DataFrameVector)
        super().__init__()

    @abstractmethod
    def __call__(self, df: pd.DataFrame, **kwargs) -> Union[pd.Series, None]:
        pass


class DataFrameComparisonVector(_Vector):
    ALL: Mapping[str, Type['DataFrameComparisonVector']] = {}

    def __init__(self):
        _register(self, DataFrameComparisonVector)
        super(DataFrameComparisonVector, self).__init__()

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[pd.Series, None]:
        pass


class ColumnMetricVector(DataFrameVector):
    def __init__(self, metric: ColumnMetric):
        self.metric = metric
        self.name = f'{metric.name}_vector'
        super(ColumnMetricVector, self).__init__()

    def __call__(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        return df.apply(func=self.metric, axis='index', raw=False, **kwargs)


class ColumnComparisonVector(DataFrameComparisonVector):
    def __init__(self, metric: ColumnComparison):
        self.metric = metric
        self.name = f'{metric.name}_vector'
        super().__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[pd.Series, None]:

        return pd.Series(
            data=[self.metric(df_old, df_new, col, **kwargs) for col in df_old.columns], index=df_old.columns,
            name=self.metric.name
        )


class RollingColumnMetricVector(ColumnVector):
    def __init__(self, metric: ColumnMetric):
        self.metric = metric
        self.name = f'rolling_{metric.name}_vector'
        super().__init__()

    def __call__(self, sr: pd.Series, window: int = 5, **kwargs) -> pd.Series:
        length = len(sr)
        pad = (window - 1) // 2
        offset = (window - 1) % 2

        sr2 = pd.Series(index=sr.index, dtype=float, name=sr.name)
        for n in range(pad, length - pad):
            sr_window = sr.iloc[n:n + 2 * pad + offset]
            sr2.iloc[n] = self.metric(sr_window, **kwargs)

        return sr2


class ChainColumnVector(ColumnVector):
    def __init__(self, *args):
        self.metrics = args
        self.name = '|'.join([m.name for m in self.metrics])
        super().__init__()

    def __call__(self, sr: pd.Series, **kwargs) -> pd.Series:
        for metric in self.metrics:
            sr = metric(sr, **kwargs)

        return sr


# Metrics that return an array instead of one value
# ----------------------------------------------------
class _Matrix(ABC):
    ALL: Mapping[str, Type['_Matrix']] = {}
    name: Union[str, None] = None
    tags: List[str] = []

    def __init__(self):
        _register(self, _Matrix)
        super(_Matrix, self).__init__()


class DataFrameMatrix(_Matrix):
    ALL: Dict[str, Type['DataFrameMatrix']] = {}

    def __init__(self):
        _register(self, DataFrameMatrix)
        super(DataFrameMatrix, self).__init__()

    @abstractmethod
    def __call__(self, df: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        pass


class DataFrameComparisonMatrix(_Matrix):
    ALL: Dict[str, Type['DataFrameMatrix']] = {}

    def __init__(self):
        _register(self, DataFrameComparisonMatrix)
        super(DataFrameComparisonMatrix, self).__init__()

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        pass


class TwoColumnComparisonMatrix(DataFrameComparisonMatrix):
    def __init__(self, metric: TwoColumnComparison):
        self.metric = metric
        self.name = f'{metric.name}_matrix'
        super(TwoColumnComparisonMatrix, self).__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        columns = df_old.columns
        matrix = pd.DataFrame(index=columns, columns=columns)

        if 'symmetric' in self.tags:
            for col_a, col_b in combinations(columns, 2):
                matrix[col_a][col_b] = matrix[col_b][col_a] = self.metric(df_old, df_new, col_a_name=col_a,
                                                                          col_b_name=col_b, **kwargs)
        else:
            for col_a, col_b in permutations(columns, 2):
                matrix[col_a][col_b] = self.metric(df_old, df_new, col_a_name=col_a, col_b_name=col_b, **kwargs)

        return matrix.astype(np.float32)


class TwoColumnMetricMatrix(DataFrameMatrix):
    def __init__(self, metric: TwoColumnMetric):
        self.metric = metric
        self.name = f'{metric.name}_matrix'
        super(TwoColumnMetricMatrix, self).__init__()

    def __call__(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        columns = df.columns
        matrix = pd.DataFrame(index=columns, columns=columns)

        if 'symmetric' in self.tags:
            for col_a, col_b in combinations(columns, 2):
                matrix[col_a][col_b] = matrix[col_b][col_a] = self.metric(df[col_a], df[col_b], **kwargs)
        else:
            for col_a, col_b in permutations(columns, 2):
                matrix[col_a][col_b] = self.metric(df[col_a], df[col_b], **kwargs)

        return matrix.astype(np.float32)
