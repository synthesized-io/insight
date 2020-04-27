"""This module contains metrics with different 'levels' of detail."""
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Union
from itertools import combinations

import pandas as pd


def _register(cls, registry):
    registry.append(cls)
    _Metric.ALL.append(cls)


class _Metric(ABC):
    name: Union[str, None] = None
    tags: List[str] = []
    ALL: List[Type['_Metric']] = []

    def __init__(self, **kwargs):
        self._value = None

    @property
    def value(self) -> Union[int, float, None]:
        return self._value

    @classmethod
    def create_composite_metric(cls, prefix: str, compute_fn):
        if prefix in ['Max', 'Min', 'Avg']:
            metric_base = POOLING_METRIC_MAP.get(cls.__base__, None)
        elif prefix in ['Diff']:
            metric_base = DIFF_METRIC_MAP.get(cls.__base__, None)
        else:
            metric_base = None

        if metric_base is None:
            raise ValueError

        return type(
            f"{prefix}{cls.__name__}",
            (metric_base,),
            {"compute": compute_fn, "name": f"{prefix} {cls.__name__}"}
        )


class ColumnMetric(_Metric):
    ALL: List[Type[_Metric]] = []

    def __init__(self, df: pd.DataFrame, col_name: str, **kwargs):
        super(ColumnMetric, self).__init__(**kwargs)
        self._value = self.compute(df=df, col_name=col_name, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(cls, ColumnMetric.ALL)

        @staticmethod
        def compute_max(df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col in df.columns:
                value = cls(df, col, **kwargs).value
                if value is not None:
                    values.append(value)
            return max(values) if len(values) > 0 else None

        @staticmethod
        def compute_min(df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col in df.columns:
                value = cls(df, col, **kwargs).value
                if value is not None:
                    values.append(value)
            return min(values) if len(values) > 0 else None

        @staticmethod
        def compute_avg(df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col in df.columns:
                value = cls(df, col, **kwargs).value
                if value is not None:
                    values.append(value)
            return sum(values)/len(values) if len(values) > 0 else None

        @staticmethod
        def compute_diff(df_old: pd.DataFrame, df_new: pd.DataFrame,
                         col_name: str, **kwargs) -> Union[int, float, None]:
            value_old = cls(df_old, col_name, **kwargs).value
            value_new = cls(df_new, col_name, **kwargs).value
            if value_old is None or value_new is None:
                return None
            else:
                return value_new - value_old

        cls.create_composite_metric("Max", compute_max)
        cls.create_composite_metric("Min", compute_min)
        cls.create_composite_metric("Avg", compute_avg)
        cls.create_composite_metric("Diff", compute_diff)

    @staticmethod
    @abstractmethod
    def compute(df: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        pass


class TwoColumnMetric(_Metric):
    ALL: List[Type[_Metric]] = []

    def __init__(self, df: pd.DataFrame, col_a_name: str, col_b_name: str, **kwargs):
        super(TwoColumnMetric, self).__init__(**kwargs)
        self._value = self.compute(df=df, col_a_name=col_a_name, col_b_name=col_b_name, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(cls, TwoColumnMetric.ALL)

        @staticmethod
        def compute_max(df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col_a, col_b in combinations(df.columns, 2):
                value = cls(df, col_a, col_b, **kwargs).value
                if value is not None:
                    values.append(value)
            return max(values) if len(values) > 0 else None

        @staticmethod
        def compute_min(df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col_a, col_b in combinations(df.columns, 2):
                value = cls(df, col_a, col_b, **kwargs).value
                if value is not None:
                    values.append(value)
            return min(values) if len(values) > 0 else None

        @staticmethod
        def compute_avg(df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col_a, col_b in combinations(df.columns, 2):
                value = cls(df, col_a, col_b, **kwargs).value
                if value is not None:
                    values.append(value)
            return sum(values) / len(values) if len(values) > 0 else None

        @staticmethod
        def compute_diff(df_old: pd.DataFrame, df_new: pd.DataFrame,
                         col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
            value_old = cls(df_old, col_a_name, col_b_name, **kwargs).value
            value_new = cls(df_new, col_a_name, col_b_name, **kwargs).value
            if value_old is None or value_new is None:
                return None
            else:
                return value_new - value_old

        cls.create_composite_metric("Max", compute_max)
        cls.create_composite_metric("Min", compute_min)
        cls.create_composite_metric("Avg", compute_avg)
        cls.create_composite_metric("Diff", compute_diff)

    @staticmethod
    @abstractmethod
    def compute(df: pd.DataFrame, col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        pass


class DataFrameMetric(_Metric):
    ALL: List[Type[_Metric]] = []

    def __init__(self, df: pd.DataFrame, *args, **kwargs):
        super(DataFrameMetric, self).__init__(df=df, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(cls, DataFrameMetric.ALL)

        @staticmethod
        def compute_diff(df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            value_old = cls(df_old, **kwargs).value
            value_new = cls(df_new, **kwargs).value
            if value_old is None or value_new is None:
                return None
            else:
                return value_new - value_old

        cls.create_composite_metric("Diff", compute_diff)

    @staticmethod
    @abstractmethod
    def compute(df: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        pass


class ColumnComparison(_Metric):
    ALL: List[Type[_Metric]] = []

    def __init__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs):
        super(ColumnComparison, self).__init__(df_old=df_old, df_new=df_new, col_name=col_name, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(cls, ColumnComparison.ALL)

        @staticmethod
        def compute_max(df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col in df_old.columns:
                value = cls(df_old, df_new, col).value
                if value is not None:
                    values.append(value)
            return max(values) if len(values) > 0 else None

        @staticmethod
        def compute_min(df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col in df_old.columns:
                value = cls(df_old, df_new, col).value
                if value is not None:
                    values.append(value)
            return min(values) if len(values) > 0 else None

        @staticmethod
        def compute_avg(df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col in df_old.columns:
                value = cls(df_old, df_new, col).value
                if value is not None:
                    values.append(value)
            return sum(values)/len(values) if len(values) > 0 else None

        cls.create_composite_metric("Max", compute_max)
        cls.create_composite_metric("Min", compute_min)
        cls.create_composite_metric("Avg", compute_avg)

    @staticmethod
    @abstractmethod
    def compute(df_old: pd.DataFrame, df_new: pd.DataFrame, col_name: str, **kwargs) -> Union[int, float, None]:
        pass


class TwoColumnComparison(_Metric):
    ALL: List[Type[_Metric]] = []

    def __init__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, col_a_name: str, col_b_name: str, **kwargs):
        super(TwoColumnComparison, self).__init__(df_old=df_old, df_new=df_new,
                                                  col_a_name=col_a_name, col_b_name=col_b_name, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(cls, TwoColumnComparison.ALL)

        @staticmethod
        def compute_max(df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col_a, col_b in combinations(df_old.columns, 2):
                value = cls(df_old, df_new, col_a, col_b).value
                if value is not None:
                    values.append(value)
            return max(values) if len(values) > 0 else None

        @staticmethod
        def compute_min(df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col_a, col_b in combinations(df_old.columns, 2):
                value = cls(df_old, df_new, col_a, col_b).value
                if value is not None:
                    values.append(value)
            return min(values) if len(values) > 0 else None

        @staticmethod
        def compute_avg(df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
            values: List[Union[int, float]] = []
            for col_a, col_b in combinations(df_old.columns, 2):
                value = cls(df_old, df_new, col_a, col_b).value
                if value is not None:
                    values.append(value)
            return sum(values) / len(values) if len(values) > 0 else None

        cls.create_composite_metric("Max", compute_max)
        cls.create_composite_metric("Min", compute_min)
        cls.create_composite_metric("Avg", compute_avg)

    @staticmethod
    @abstractmethod
    def compute(df_old: pd.DataFrame, df_new: pd.DataFrame,
                col_a_name: str, col_b_name: str, **kwargs) -> Union[int, float, None]:
        pass


class DataFrameComparison(_Metric):
    ALL: List[Type[_Metric]] = []

    def __init__(self, df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs):
        super(DataFrameComparison, self).__init__(df_old=df_old, df_new=df_new, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register(cls, DataFrameComparison.ALL)

    @staticmethod
    @abstractmethod
    def compute(df_old: pd.DataFrame, df_new: pd.DataFrame, **kwargs) -> Union[int, float, None]:
        pass


POOLING_METRIC_MAP: Dict[Type, Type[_Metric]] = {
    ColumnMetric: DataFrameMetric,
    TwoColumnMetric: DataFrameMetric,
    ColumnComparison: DataFrameComparison,
    TwoColumnComparison: DataFrameComparison
}

DIFF_METRIC_MAP: Dict[Type, Type[_Metric]] = {
    ColumnMetric: ColumnComparison,
    TwoColumnMetric: TwoColumnComparison,
    DataFrameMetric: DataFrameComparison
}
