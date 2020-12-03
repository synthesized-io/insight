from abc import abstractmethod
from typing import Any, Generic, Optional, TypeVar

import pandas as pd

from .domain import Domain
from .value_meta import Nominal, Affine, NType, AType

ModelType = TypeVar('ModelType', bound='Model')


class Model:

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def fit(self: ModelType, df: pd.DataFrame) -> ModelType:
        self._fitted = True
        return self

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def probability(self, x: Any) -> float:
        """Get the probability mass of the category x."""
        pass


DiscreteModelType = TypeVar('DiscreteModelType', bound='DiscreteModel[Any]')


class DiscreteModel(Nominal[NType], Model, Generic[NType]):
    class_name = "DiscreteModel"

    def __init__(self, name: str, domain: Optional[Domain[NType]] = None, nan_freq: Optional[float] = None):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq)  # type: ignore

    def fit(self: DiscreteModelType, df: pd.DataFrame) -> DiscreteModelType:
        super().fit(df=df)
        if not self._extracted:
            self.extract(df=df)
        return self

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def probability(self, x: Any) -> float:
        pass


ContinuousModelType = TypeVar('ContinuousModelType', bound='ContinuousModel[Any]')


class ContinuousModel(Affine[AType], Model, Generic[AType]):
    class_name = "ContinuousModel"

    def __init__(
            self, name: str, domain: Optional[Domain[NType]] = None, nan_freq: Optional[float] = None,
            min: Optional[AType] = None, max: Optional[AType] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)  # type: ignore

    def fit(self: ContinuousModelType, df: pd.DataFrame) -> ContinuousModelType:
        super().fit(df=df)
        if not self._extracted:
            self.extract(df=df)
        return self

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def probability(self, x: Any) -> float:
        pass
