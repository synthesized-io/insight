from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Sequence, TypeVar  # noqa: F401 (flake8 doesn't like the Any import here)

import numpy as np
import pandas as pd

from .value_meta import Affine, AType, Meta, Nominal, NType

ModelType = TypeVar('ModelType', bound='Model')
ContinuousModelType = TypeVar('ContinuousModelType', bound='ContinuousModel[Any]')
DiscreteModelType = TypeVar('DiscreteModelType', bound='DiscreteModel[Any]')


class Model(ABC):

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'Model':
        """Extract the children of this Meta."""
        children = getattr(self, 'children', [])

        assert isinstance(self, Meta)
        with self.unfold(df) as sub_df:
            for child in children:
                child.fit(sub_df)
        self._fitted = True
        return self

    @abstractmethod
    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        pass

    def add_nans(self, sr: pd.Series, nan_freq: Optional[float]) -> pd.DataFrame:
        if nan_freq and nan_freq > 0:
            sr = np.where(sr.index.isin(np.random.choice(sr.index, size=int(len(sr) * nan_freq))), np.nan, sr)
        return sr


class DiscreteModel(Nominal[NType], Model, Generic[NType]):

    def __init__(self, name: str, categories: Optional[Sequence[NType]] = None,
                 nan_freq: Optional[float] = None, num_rows: Optional[int] = None):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)  # type: ignore

    def fit(self: DiscreteModelType, df: pd.DataFrame) -> DiscreteModelType:
        super().fit(df=df)
        if not self._extracted:
            self.extract(df=df)
        return self


class ContinuousModel(Affine[AType], Model, Generic[AType]):

    def __init__(
            self, name: str, categories: Optional[Sequence[AType]] = None, nan_freq: Optional[float] = None,
            min: Optional[AType] = None, max: Optional[AType] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, min=min, max=max)  # type: ignore

    def fit(self: ContinuousModelType, df: pd.DataFrame) -> ContinuousModelType:
        super().fit(df=df)
        if not self._extracted:
            self.extract(df=df)
        return self
