from abc import abstractmethod
from typing import Any, Generic, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd

from .value_meta import Affine, AType, Nominal, NType

ModelType = TypeVar('ModelType', bound='Model')
ContinuousModelType = TypeVar('ContinuousModelType', bound='ContinuousModel[Any]')
DiscreteModelType = TypeVar('DiscreteModelType', bound='DiscreteModel[Any]')


class Model:

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def fit(self: ModelType, df: pd.DataFrame) -> ModelType:
        self._fitted = True
        return self

    @abstractmethod
    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def probability(self, x: Any) -> float:
        """Get the probability mass of the category x."""
        raise NotImplementedError

    def add_nans(self, sr: pd.Series, nan_freq: Optional[float]) -> pd.DataFrame:
        if nan_freq and nan_freq > 0:
            sr = np.where(sr.index.isin(np.random.choice(sr.index, size=int(len(sr) * 0.1))), np.nan, sr)
        return sr


class DiscreteModel(Nominal[NType], Model, Generic[NType]):

    def __init__(self, name: str, categories: Optional[Sequence[NType]] = None, nan_freq: Optional[float] = None):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore

    def fit(self: DiscreteModelType, df: pd.DataFrame) -> DiscreteModelType:
        super().fit(df=df)
        if not self._extracted:
            self.extract(df=df)
        return self

    @abstractmethod
    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:
        pass

    @abstractmethod
    def probability(self, x: Any) -> float:
        pass


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

    @abstractmethod
    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:
        pass

    @abstractmethod
    def probability(self, x: Any) -> float:
        pass
