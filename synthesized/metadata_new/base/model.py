from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Sequence

import pandas as pd

from .value_meta import Nominal, Affine, NType, AType

ModelType = TypeVar('ModelType', bound='Model')
ContinuousModelType = TypeVar('ContinuousModelType', bound='ContinuousModel[Any]')
DiscreteModelType = TypeVar('DiscreteModelType', bound='DiscreteModel[Any]')


class Model:

    _model_registry: Dict[str, Type['Model']] = {}

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def __init_subclass__(cls: Type[ModelType]) -> None:
        super().__init_subclass__()
        Model._model_registry[cls.__name__] = cls

    def fit(self: ModelType, df: pd.DataFrame) -> ModelType:
        self._fitted = True
        return self

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def probability(self, x: Any) -> float:
        """Get the probability mass of the category x."""
        raise NotImplementedError


class DiscreteModel(Nominal[NType], Model, Generic[NType]):

    def __init__(self, name: str, categories: Optional[Sequence[NType]] = None, nan_freq: Optional[float] = None):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore

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
    def sample(self, n: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def probability(self, x: Any) -> float:
        pass
