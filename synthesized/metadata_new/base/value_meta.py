from abc import abstractmethod
from typing import Any, Generic, TypeVar, Optional, Dict, Sequence
from functools import cmp_to_key

import numpy as np
import pandas as pd

from .meta import Meta

DType = TypeVar('DType', covariant=True)
NType = TypeVar("NType", str, np.datetime64, np.timedelta64, int, float, bool, covariant=True)
OType = TypeVar("OType", str, np.datetime64, np.timedelta64, int, float, bool, covariant=True)
AType = TypeVar("AType", np.datetime64, np.timedelta64, int, float, bool, covariant=True)
SType = TypeVar("SType", np.timedelta64, int, float, bool, covariant=True)
RType = TypeVar("RType", float, bool, covariant=True)

ValueMetaType = TypeVar('ValueMetaType', bound='ValueMeta[Any]')
NominalType = TypeVar('NominalType', bound='Nominal[Any]')
OrdinalType = TypeVar('OrdinalType', bound='Ordinal[Any]')
AffineType = TypeVar('AffineType', bound='Affine[Any]')
ScaleType = TypeVar('ScaleType', bound='Scale[Any]')
RingType = TypeVar('RingType', bound='Ring[Any]')


class ValueMeta(Meta, Generic[DType]):
    """
    Base class for meta information that describes a pandas series.

    ValueMeta objects act as leaf nodes in the Meta hierarchy. Derived
    classes must implement extract. All attributes should be set in extract.

    Attributes:
        name: The pd.Series name that this ValueMeta describes.
        dtype: Optional; The numpy dtype that this meta describes.
    """
    dtype: Optional[str] = None

    def __init__(self, name: str):
        super().__init__(name=name)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, dtype={self.dtype})'


class Nominal(ValueMeta[NType], Generic[NType]):
    """
    Nominal meta.

    Nominal describes any data that can be categorised, but has
    no quantitative interpretation, i.e it can only be given a name.
    Nominal data cannot be ordered nor compared, and there is no notion
    of 'closeness', e.g blood types ['AA', 'B', 'AB', 'O'].

    Attributes:
        domain: Optional; list of category names.
    """

    def __init__(
            self, name: str, categories: Optional[Sequence[NType]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name)
        self.categories: Optional[Sequence[NType]] = categories
        self.nan_freq = nan_freq

    def extract(self: NominalType, df: pd.DataFrame) -> NominalType:
        """Extract the domain and their relative frequencies from a data frame, if not already set."""
        super().extract(df)
        if self.categories is None:
            self.categories = [c for c in np.array(df[self.name].unique(), dtype=self.dtype)]

        if self.nan_freq is None:
            self.nan_freq = df[self.name].isna().sum() / len(df)

        return self

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "nan_freq": self.nan_freq,
            "categories": [c for c in self.categories] if self.categories is not None else None
        })

        return d


class Ordinal(Nominal[OType], Generic[OType]):
    """
    Ordinal meta.

    Ordinal meta describes categorical data that can be compared and ordered,
    and data can be arranged on a relative scale, e.g
    the Nandos scale: ['extra mild', 'mild', 'medium', 'hot', 'extra hot']

    Attributes:
        min: Optional; the minimum category
        max: Optional; the maximum category
    """

    def __init__(
            self, name: str, categories: Optional[Sequence[OType]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore
        self._min = None
        self._max = None

    def extract(self: OrdinalType, df: pd.DataFrame) -> OrdinalType:
        super().extract(df)
        assert self.categories is not None
        self.categories = self.sort(self.categories)

        self._min = self.categories[0]
        self._max = self.categories[-1]

        return self

    @property
    def min(self) -> Optional[OType]:
        return self._min

    @property
    def max(self) -> Optional[OType]:
        return self._max

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()

        return d

    def less_than(self, x: OType, y: OType) -> bool:

        b: bool = x < y
        return b

    def _predicate(self, x: Any, y: Any) -> int:
        if self.less_than(x, y):
            return 1
        elif x == y:
            return 0
        else:
            return -1

    def sort(self, sr: Sequence[OType]) -> Sequence[OType]:
        """Sort pd.Series according to the ordering of this meta"""
        key = cmp_to_key(self._predicate)
        return sorted(sr, key=key, reverse=True)


class Affine(Ordinal[AType], Generic[AType]):
    """
    Affine meta.

    Affine describes data that can live in an affine space, where
    there is a notion of relative distance between points. Here, only
    the operation of subtraction is permitted between data points. Addition
    of points is not valid, however relative differences can be added to points.
    e.g Given two dates, they can be subtracted to get a relative time delta, but
    it makes no sense to add them.

    Attributes:

    """

    def __init__(
            self, name: str, categories: Optional[Sequence[AType]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore

    def extract(self: AffineType, df: pd.DataFrame) -> AffineType:
        super().extract(df)
        return self

    @property
    @abstractmethod
    def unit_meta(self: AffineType) -> 'Scale[Any]':
        pass


class Scale(Affine[SType], Generic[SType]):
    """
    Scale meta.

    Scale describes data that can live in a vector space. Data points
    can be reached through linear combinations of other points,
    and points can also be multiplied by scalars.

    This describes pretty much most continuous types.

    Attributes:
    """
    precision: SType

    def __init__(
            self, name: str, categories: Optional[Sequence[SType]] = None, nan_freq: Optional[float] = None,
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore

    def extract(self: ScaleType, df: pd.DataFrame) -> ScaleType:
        super().extract(df)

        return self

    @property
    def unit_meta(self: ScaleType) -> ScaleType:
        return self


class Ring(Scale[RType], Generic[RType]):

    def __init__(
            self, name: str, categories: Optional[Sequence[RType]] = None, nan_freq: Optional[float] = None,
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore

    def extract(self: RingType, df: pd.DataFrame) -> RingType:
        super().extract(df)
        return self
