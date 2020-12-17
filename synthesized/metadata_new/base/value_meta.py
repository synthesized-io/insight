from abc import abstractmethod
from typing import Any, Generic, TypeVar, Optional, Dict, MutableSequence
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
    class_name: str = 'ValueMeta'
    dtype: Optional[str] = None

    def __init__(self, name: str):
        super().__init__(name=name)

    def __repr__(self) -> str:
        return f'{self.class_name}(name={self.name}, dtype={self.dtype})'


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
    class_name: str = 'Nominal'

    def __init__(
            self, name: str, categories: Optional[MutableSequence[NType]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name)
        self.categories: Optional[MutableSequence[NType]] = categories
        self.nan_freq = nan_freq

    def extract(self: NominalType, df: pd.DataFrame) -> NominalType:
        """Extract the domain and their relative frequencies from a data frame, if not already set."""
        super().extract(df)
        if self.categories is None:
            self.categories = [x for x in df[self.name].unique()]

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
    class_name: str = 'Ordinal'

    def __init__(
            self, name: str, categories: Optional[MutableSequence[OType]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore
        self._min = None
        self._max = None

    def extract(self: OrdinalType, df: pd.DataFrame) -> OrdinalType:
        super().extract(df)
        self.categories = self.sort(self.categories)

        self._min = self.categories[0]
        self._max = self.categories[-1]

        return self

    @property
    def min(self) -> Optional[OType]:
        return self._min

    @min.setter
    def min(self, x: Optional[OType]) -> None:
        self._min = x

        if self._min is None:
            return

        n = 0
        for n, value in enumerate(self.categories):
            if self.less_than(x, value):
                break

        self.categories = self.categories[n:]

        if x != self.categories[0]:
            self.categories.insert(0, x)

        if self._max is not None and not self.less_than(self._min, self._max):
            self._max = None

    @property
    def max(self) -> Optional[OType]:
        return self.categories[-1] if self.categories is not None else None

    @max.setter
    def max(self, x: Optional[OType]) -> None:
        self._max = x

        if x is None:
            return

        n = 0
        for n, value in enumerate(reversed(self.categories)):
            if self.less_than(value, x):
                break

        self.categories = self.categories[:-n]

        if x != self.categories[-1]:
            self.categories.append(x)

        if self._min is not None and not self.less_than(self._min, self._max):
            self._min = None

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

    def sort(self, sr: MutableSequence[OType]) -> MutableSequence[OType]:
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
    class_name: str = 'Affine'

    def __init__(
            self, name: str, categories: Optional[MutableSequence[AType]] = None, nan_freq: Optional[float] = None
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
    class_name: str = 'Scale'
    precision: SType

    def __init__(
            self, name: str, categories: Optional[MutableSequence[SType]] = None, nan_freq: Optional[float] = None,
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore

    def extract(self: ScaleType, df: pd.DataFrame) -> ScaleType:
        super().extract(df)

        return self

    @property
    def unit_meta(self: ScaleType) -> ScaleType:
        return self


class Ring(Scale[RType], Generic[RType]):
    class_name: str = 'Ring'

    def __init__(
            self, name: str, categories: Optional[MutableSequence[RType]] = None, nan_freq: Optional[float] = None,
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore

    def extract(self: RingType, df: pd.DataFrame) -> RingType:
        super().extract(df)
        return self
