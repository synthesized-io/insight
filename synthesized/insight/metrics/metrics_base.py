"""This module contains metrics with different 'levels' of detail."""
from abc import ABC, abstractmethod
from itertools import combinations, permutations
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from ...metadata.factory import MetaExtractor
from ...model import DataFrameModel, factory


class _Metric(ABC):
    name: Optional[str] = None
    tags: List[str] = []

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.name}"


class ColumnMetric(_Metric):

    def _extract_models(self, sr: pd.Series, df_model: Optional[DataFrameModel] = None) -> DataFrameModel:
        """Method for extracting models from dataframe if not already extracted"""
        if df_model is None:
            df = pd.DataFrame(data={sr.name: sr})
            df_meta = MetaExtractor.extract(df)
            df_model = factory.ModelFactory()(df_meta)

        return df_model

    @abstractmethod
    def __call__(self, sr: pd.Series, df_model: Optional[DataFrameModel] = None) -> Union[int, float, None]:
        pass


class TwoColumnMetric(_Metric):

    def _extract_models(self, sr_a: pd.Series, sr_b: pd.Series,
                        df_model: Optional[DataFrameModel] = None) -> DataFrameModel:
        """Method for extracting models from dataframe if not already extracted"""
        if df_model is None:
            df = pd.DataFrame(data={sr_a.name: sr_a, sr_b.name: sr_b})
            df_meta = MetaExtractor.extract(df)
            df_model = factory.ModelFactory()(df_meta)

        return df_model

    @abstractmethod
    def __call__(self, sr_a: pd.Series, sr_b: pd.Series,
                 df_model: Optional[DataFrameModel] = None) -> Union[int, float, None]:
        pass


class DataFrameMetric(_Metric):

    @abstractmethod
    def __call__(self, df: pd.DataFrame, df_model: Optional[DataFrameModel] = None) -> Union[int, float, None]:
        pass


class TwoDataFrameMetric(_Metric):

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame,
                 df_model: Optional[DataFrameModel] = None) -> Union[int, float, None]:
        pass


# Adapters to transform Metrics into Composite Metrics
# ----------------------------------------------------

class DiffColumnMetricAdapter(TwoColumnMetric):
    def __init__(self, metric: ColumnMetric):
        self.metric = metric
        self.name = f'diff_{metric.name}'
        super().__init__()

    def __call__(self, sr_a: pd.Series, sr_b: pd.Series,
                 df_model: Optional[DataFrameModel] = None) -> Union[int, float, None]:
        value_old = self.metric(sr_a, df_model=df_model)
        value_new = self.metric(sr_b, df_model=df_model)
        if value_old is None or value_new is None:
            return None
        else:
            return value_new - value_old


# Vectors that return a series instead of one value
# ----------------------------------------------------
class _Vector(ABC):
    name: Union[str, None] = None
    tags: List[str] = []


class ColumnVector(_Vector):

    @abstractmethod
    def __call__(self, sr: pd.Series, df_model: Optional[DataFrameModel] = None) -> Union[pd.Series, None]:
        pass


class DataFrameVector(_Vector):

    @abstractmethod
    def __call__(self, df: pd.DataFrame, df_model: Optional[DataFrameModel] = None) -> Union[pd.Series, None]:
        pass


class TwoDataFrameVector(_Vector):

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame,
                 df_model: Optional[DataFrameModel] = None) -> Union[pd.Series, None]:
        pass


class ColumnMetricVector(DataFrameVector):
    def __init__(self, metric: ColumnMetric):
        self.metric = metric
        self.name = f'{metric.name}_vector'
        super(ColumnMetricVector, self).__init__()

    def __call__(self, df: pd.DataFrame, df_model: Optional[DataFrameModel] = None) -> Union[pd.Series, None]:
        if df is None:
            return None
        return df.apply(func=self.metric, axis='index', raw=False, df_model=df_model)


class ColumnComparisonVector(TwoDataFrameVector):
    def __init__(self, metric: TwoColumnMetric):
        self.metric = metric
        self.name = f'{metric.name}_vector'
        super().__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame,
                 df_model: Optional[DataFrameModel] = None) -> Union[pd.Series, None]:
        if df_old is None or df_new is None:
            return None
        return pd.Series(
            data=[self.metric(df_old[col], df_new[col], df_model=df_model) for col in df_old.columns],
            index=df_old.columns,
            name=self.metric.name
        )


class RollingColumnMetricVector(ColumnVector):
    def __init__(self, metric: ColumnMetric, window: int = 5):
        self.metric = metric
        self.window = window
        self.name = f'rolling_{metric.name}_vector'
        super().__init__()

    def __call__(self, sr: pd.Series, df_model: Optional[DataFrameModel] = None) -> Union[pd.Series, None]:
        length = len(sr)
        pad = (self.window - 1) // 2
        offset = (self.window - 1) % 2

        sr2 = pd.Series(index=sr.index, dtype=float, name=sr.name)
        for n in range(pad, length - pad):
            sr_window = sr.iloc[n:n + 2 * pad + offset]
            sr2.iloc[n] = self.metric(sr_window, df_model=df_model)

        return sr2


class ChainColumnVector(ColumnVector):
    def __init__(self, *args):
        self.metrics = args
        self.name = '|'.join([m.name for m in self.metrics])
        super().__init__()

    def __call__(self, sr: pd.Series, df_model: Optional[DataFrameModel] = None) -> Union[pd.Series, None]:

        for metric in self.metrics:
            sr = metric(sr, df_model=df_model)

        return sr


# Metrics that return an array instead of one value
# ----------------------------------------------------
class _Matrix(ABC):
    name: Union[str, None] = None
    tags: List[str] = []


class DataFrameMatrix(_Matrix):

    @abstractmethod
    def __call__(self, df: pd.DataFrame, df_model: Optional[DataFrameModel] = None) -> Union[pd.DataFrame, None]:
        pass


class TwoDataFrameMatrix(_Matrix):

    @abstractmethod
    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame,
                 df_model: Optional[DataFrameModel] = None) -> Union[pd.DataFrame, None]:
        pass


class TwoColumnMetricMatrix(DataFrameMatrix):
    def __init__(self, metric: TwoColumnMetric):
        self.metric = metric
        self.name = f'{metric.name}_matrix'
        super(TwoColumnMetricMatrix, self).__init__()

    def __call__(self, df: pd.DataFrame, df_model: Optional[DataFrameModel] = None) -> Union[pd.DataFrame, None]:
        columns = df.columns
        matrix = pd.DataFrame(index=columns, columns=columns)

        if 'symmetric' in self.tags:
            for col_a, col_b in combinations(columns, 2):
                matrix[col_a][col_b] = matrix[col_b][col_a] = self.metric(df[col_a], df[col_b], df_model=df_model)
        else:
            for col_a, col_b in permutations(columns, 2):
                matrix[col_a][col_b] = self.metric(df[col_a], df[col_b], df_model=df_model)

        return matrix.astype(np.float32)


class DiffMetricMatrix(TwoDataFrameMatrix):
    def __init__(self, metric: DataFrameMatrix):
        self.metric = metric
        self.name = f'diff_{metric.name}'
        super().__init__()

    def __call__(self, df_old: pd.DataFrame, df_new: pd.DataFrame,
                 df_model: Optional[DataFrameModel] = None) -> Union[pd.DataFrame, None]:
        if df_old is None or df_new is None:
            return None

        matrix_old = self.metric(df=df_old, df_model=df_model)
        matrix_new = self.metric(df=df_new, df_model=df_model)

        if matrix_old is None or matrix_new is None:
            return None

        return matrix_new - matrix_old


class ModellingMetric(_Metric):

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> Union[float, None]:
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
