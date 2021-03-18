from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, Sequence, Type, TypeVar, cast

import numpy as np
import pandas as pd

from ..metadata import AffineType, AType, Meta, MetaType, NominalType, NType
from ..util import get_all_subclasses

ModelType = TypeVar('ModelType', bound='Model')
ContinuousModelType = TypeVar('ContinuousModelType', bound='ContinuousModel')
DiscreteModelType = TypeVar('DiscreteModelType', bound='DiscreteModel')


class Model(Generic[MetaType]):

    def __init__(self, meta: MetaType) -> None:
        self._meta = meta
        self._fitted = False

    @property
    def meta(self) -> MetaType:
        return self._meta

    @property
    def name(self):
        return self._meta.name

    @property
    def num_rows(self) -> Optional[int]:
        return self._meta.num_rows

    def fit(self, df: pd.DataFrame) -> 'Model':
        """Extract the children of this Meta."""
        if not self._meta._extracted:
            self._meta.extract(df=df)
        self._fitted = True

        return self

    @abstractmethod
    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        pass

    def add_nans(self, sr: pd.Series, nan_freq: Optional[float]) -> pd.DataFrame:
        if nan_freq and nan_freq > 0:
            sr = np.where(sr.index.isin(np.random.choice(sr.index, size=int(len(sr) * nan_freq))), np.nan, sr)

        return sr

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "class_name": self.__class__.__name__,
            "meta": self._meta.to_dict(),
            "fitted": self._fitted
        }

        return d

    @classmethod
    def from_dict(cls: Type[ModelType], d: Dict[str, object]) -> ModelType:
        meta_dict = cast(Dict[str, object], d["meta"])
        meta = Meta[Any].from_dict(meta_dict)
        model = cls(meta=meta)
        model._fitted = cast(bool, d["fitted"])

        return model

    @classmethod
    def from_dict_with_class_name(cls: Type[ModelType], d: Dict[str, object]) -> ModelType:
        registy = cls.get_registry()
        class_name = cast(str, d["class_name"])
        if class_name not in registy.keys():
            raise ValueError(f"Given meta {class_name} not found in Meta subclasses.")
        model = registy[class_name].from_dict(d)
        return model

    @classmethod
    def get_registry(cls: Type[ModelType]) -> Dict[str, Type[ModelType]]:
        return {sc.__name__: sc for sc in get_all_subclasses(cls)}


class DiscreteModel(Model[NominalType], Generic[NominalType, NType]):

    def __init__(self, meta: NominalType):
        super().__init__(meta=meta)

    @property
    def dtype(self) -> str:
        return self._meta.dtype

    @property
    @abstractmethod
    def categories(self) -> Sequence[NType]:
        pass

    @property
    def nan_freq(self) -> Optional[float]:
        return self._meta.nan_freq


class ContinuousModel(Model[AffineType], Generic[AffineType, AType]):

    def __init__(self, meta: AffineType):
        super().__init__(meta=meta)

    @property
    def dtype(self) -> str:
        return self._meta.dtype

    @property
    @abstractmethod
    def min(self) -> Optional[AType]:
        pass

    @property
    @abstractmethod
    def max(self) -> Optional[AType]:
        pass

    @property
    def nan_freq(self) -> Optional[float]:
        return self._meta.nan_freq
