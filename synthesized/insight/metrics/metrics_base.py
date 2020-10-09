"""This module contains metrics with different 'levels' of detail."""
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union
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


class _Metric:
    ALL: Mapping[str, '_Metric'] = {}
    name: Union[str, None] = None
    tags: List[str] = []

    def __init__(self):
        _register(self, _Metric)
        super(_Metric, self).__init__()

    def __call__(self, **kwargs) -> Union[int, float, None]:
        pass


class ColumnMetric(_Metric):
    ALL: Dict[str, 'ColumnMetric'] = {}

    def __init__(self):
        _register(self, ColumnMetric)
        DiffColumnMetricAdapter(self)
        ColumnMetricVector(self)

        super(ColumnMetric, self).__init__()

    @abstractmethod
    def __call__(self, sr: pd.Series = None, **kwargs) -> Union[int, float, None]:
        pass


class TwoColumnMetric(_Metric):
    ALL: Dict[str, 'TwoColumnMetric'] = {}

    def __init__(self):
        _register(self, TwoColumnMetric)

        TwoColumnMetricMatrix(self)

        super(TwoColumnMetric, self).__init__()

    def check_column_types(self, sr_a: pd.Series, sr_b: pd.Series, **kwargs):
        dp_or_df = kwargs.get('dp')
        if dp_or_df is None:
            dp_or_df = pd.DataFrame([sr_a, sr_b])

        if "nominal" in self.tags:
            categorical, _ = categorical_or_continuous_values(dp_or_df)
            categorical_columns = [v.name for v in categorical]

            if sr_a.name not in categorical_columns or sr_b.name not in categorical_columns:
                return False

        if "ordinal" in self.tags:
            _, continuous = categorical_or_continuous_values(dp_or_df)
            continuous_columns = [v.name for v in continuous]

            if sr_a.name not in continuous_columns or sr_b.name not in continuous_columns:
                return False

        if kwargs.get('continuous_input_only', False):
            _, continuous = categorical_or_continuous_values(dp_or_df)
            continuous_columns = [v.name for v in continuous]

            if sr_a.name not in continuous_columns:
                return False

        if kwargs.get('categorical_output_only', False):
            categorical, _ = categorical_or_continuous_values(dp_or_df)
            categorical_columns = [v.name for v in categorical]

            if sr_b.name not in categorical_columns:
                return False

        return True

    @abstractmethod
    def __call__(self, sr_a: pd.Series = None, sr_b: pd.Series = None, **kwargs) -> Union[int, float, None]:
        pass


class DataFrameMetric(_Metric):
    ALL: Dict[str, 'DataFrameMetric'] = {}

    def __init__(self):
        _register(self, DataFrameMetric)

        super(DataFrameMetric, self).__init__()

    @abstractmethod
    def __call__(self, df: pd.DataFrame = None, **kwargs) -> Union[int, float, None]:
        pass


class TwoDataFrameMetric(_Metric):
    ALL: Dict[str, 'TwoDataFrameMetric'] = {}

    def __init__(self):
        _register(self, TwoDataFrameMetric)

        super(TwoDataFrameMetric, self).__init__()

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame = None, df_new: pd.DataFrame = None, **kwargs) -> Union[int, float, None]:
        pass


# Adapters to transform Metrics into Composite Metrics
# ----------------------------------------------------

class DiffColumnMetricAdapter(TwoColumnMetric):
    def __init__(self, metric: ColumnMetric):
        self.metric = metric
        self.name = f'diff_{metric.name}'
        super().__init__()

    def __call__(self, sr_a: pd.Series = None, sr_b: pd.Series = None, **kwargs) -> Union[int, float, None]:
        value_old = self.metric(sr_a, **kwargs)
        value_new = self.metric(sr_b, **kwargs)
        if value_old is None or value_new is None:
            return None
        else:
            return value_new - value_old


# Vectors that return a series instead of one value
# ----------------------------------------------------
class _Vector:
    ALL: Mapping[str, '_Vector'] = {}
    name: Union[str, None] = None
    tags: List[str] = []

    def __init__(self):
        _register(self, _Vector)
        super(_Vector, self).__init__()

    @abstractmethod
    def __call__(self, **kwargs) -> pd.Series:
        pass


class ColumnVector(_Vector):
    ALL: Mapping[str, 'ColumnVector'] = {}

    def __init__(self):
        _register(self, ColumnVector)
        super().__init__()

    @abstractmethod
    def __call__(self, sr: pd.Series = None, **kwargs) -> Union[pd.Series, None]:
        pass


class DataFrameVector(_Vector):
    ALL: Mapping[str, 'DataFrameVector'] = {}

    def __init__(self):
        _register(self, DataFrameVector)
        super().__init__()

    @abstractmethod
    def __call__(self, df: pd.DataFrame = None, **kwargs) -> Union[pd.Series, None]:
        pass


class TwoDataFrameVector(_Vector):
    ALL: Mapping[str, 'TwoDataFrameVector'] = {}

    def __init__(self):
        _register(self, TwoDataFrameVector)
        super(TwoDataFrameVector, self).__init__()

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame = None, df_new: pd.DataFrame = None, **kwargs) -> Union[pd.Series, None]:
        pass


class ColumnMetricVector(DataFrameVector):
    def __init__(self, metric: ColumnMetric):
        self.metric = metric
        self.name = f'{metric.name}_vector'
        super(ColumnMetricVector, self).__init__()

    def __call__(self, df: pd.DataFrame = None, **kwargs) -> Union[pd.Series, None]:
        if df is None:
            return None
        return df.apply(func=self.metric, axis='index', raw=False, **kwargs)


class ColumnComparisonVector(TwoDataFrameVector):
    def __init__(self, metric: TwoColumnMetric):
        self.metric = metric
        self.name = f'{metric.name}_vector'
        super().__init__()

    def __call__(self, df_old: pd.DataFrame = None, df_new: pd.DataFrame = None, **kwargs) -> Union[pd.Series, None]:
        if df_old is None or df_new is None:
            return None
        return pd.Series(
            data=[self.metric(df_old[col], df_new[col], **kwargs) for col in df_old.columns], index=df_old.columns,
            name=self.metric.name
        )


class RollingColumnMetricVector(ColumnVector):
    def __init__(self, metric: ColumnMetric):
        self.metric = metric
        self.name = f'rolling_{metric.name}_vector'
        super().__init__()

    def __call__(self, sr: pd.Series = None, window: int = 5, **kwargs) -> Union[pd.Series, None]:
        if sr is None:
            return None

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

    def __call__(self, sr: pd.Series = None, **kwargs) -> Union[pd.Series, None]:
        if sr is None:
            return None

        for metric in self.metrics:
            sr = metric(sr, **kwargs)

        return sr


# Metrics that return an array instead of one value
# ----------------------------------------------------
class _Matrix:
    ALL: Mapping[str, '_Matrix'] = {}
    name: Union[str, None] = None
    tags: List[str] = []

    def __init__(self):
        _register(self, _Matrix)
        super(_Matrix, self).__init__()

    @abstractmethod
    def __call__(self, **kwargs) -> Union[pd.DataFrame, None]:
        pass


class DataFrameMatrix(_Matrix):
    ALL: Dict[str, 'DataFrameMatrix'] = {}

    def __init__(self):
        _register(self, DataFrameMatrix)
        super(DataFrameMatrix, self).__init__()

    @abstractmethod
    def __call__(self, df: pd.DataFrame = None, **kwargs) -> Union[pd.DataFrame, None]:
        pass


class TwoDataFrameMatrix(_Matrix):
    ALL: Dict[str, 'DataFrameMatrix'] = {}

    def __init__(self):
        _register(self, TwoDataFrameMatrix)
        super(TwoDataFrameMatrix, self).__init__()

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame = None, df_new: pd.DataFrame = None, **kwargs) -> Union[pd.DataFrame, None]:
        pass


class TwoColumnMetricMatrix(DataFrameMatrix):
    def __init__(self, metric: TwoColumnMetric):
        self.metric = metric
        self.name = f'{metric.name}_matrix'
        super(TwoColumnMetricMatrix, self).__init__()

    def __call__(self, df: pd.DataFrame = None, **kwargs) -> Union[pd.DataFrame, None]:
        if df is None:
            return None

        columns = df.columns
        matrix = pd.DataFrame(index=columns, columns=columns)

        if 'symmetric' in self.tags:
            for col_a, col_b in combinations(columns, 2):
                matrix[col_a][col_b] = matrix[col_b][col_a] = self.metric(df[col_a], df[col_b], **kwargs)
        else:
            for col_a, col_b in permutations(columns, 2):
                matrix[col_a][col_b] = self.metric(df[col_a], df[col_b], **kwargs)

        return matrix.astype(np.float32)


class DiffMetricMatrix(TwoDataFrameMatrix):
    def __init__(self, metric: DataFrameMatrix):
        self.metric = metric
        self.name = f'diff_{metric.name}'
        super().__init__()

    def __call__(self, df_old: pd.DataFrame = None, df_new: pd.DataFrame = None, **kwargs) -> Union[pd.DataFrame, None]:
        if df_old is None or df_new is None:
            return None

        matrix_old = self.metric(df=df_old, **kwargs)
        matrix_new = self.metric(df=df_new, **kwargs)

        if matrix_old is None or matrix_new is None:
            return None

        return matrix_new - matrix_old


class ModellingMetric(_Metric):

    def __init__(self):
        _register(self, ModellingMetric)
        super(ModellingMetric, self).__init__()

    @abstractmethod
    def __call__(self, y_true: np.ndarray = None, y_pred: Optional[np.ndarray] = None, **kwargs) -> Union[float, None]:
        pass


class ClassificationMetric(ModellingMetric):
    ALL: Dict[str, 'ClassificationMetric'] = {}
    tags = ["modelling", "classification"]
    plot = False

    def __init__(self):
        _register(self, ClassificationMetric)
        super(ClassificationMetric, self).__init__()

    @abstractmethod
    def __call__(self, y_true: np.ndarray = None, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None, **kwargs) -> Union[float]:
        pass


class ClassificationPlotMetric(ModellingMetric):
    ALL: Dict[str, 'ClassificationPlotMetric'] = {}
    tags = ["modelling", "classification", "plot"]
    plot = True

    def __init__(self):
        _register(self, ClassificationPlotMetric)
        super(ClassificationPlotMetric, self).__init__()

    @abstractmethod
    def __call__(self, y_true: np.ndarray = None, y_pred: Optional[np.ndarray] = None,
                 y_pred_proba: Optional[np.ndarray] = None, **kwargs) -> Union[Any]:
        pass


class RegressionMetric(ModellingMetric):
    ALL: Dict[str, 'RegressionMetric'] = {}
    tags = ["modelling", "regression"]

    def __init__(self):
        _register(self, RegressionMetric)
        super(RegressionMetric, self).__init__()

    @abstractmethod
    def __call__(self, y_true: np.ndarray = None, y_pred: np.ndarray = None, **kwargs) -> Union[float]:
        pass
