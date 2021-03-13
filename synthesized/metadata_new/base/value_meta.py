from functools import cmp_to_key
from typing import Any, Dict, Generic, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd

from .meta import Meta

DType = TypeVar('DType', covariant=True)
NType = TypeVar("NType", str, np.datetime64, np.timedelta64, np.int64, np.float64, np.bool8, covariant=True)
OType = TypeVar("OType", str, np.datetime64, np.timedelta64, np.int64, np.float64, np.bool8, covariant=True)
AType = TypeVar("AType", np.datetime64, np.timedelta64, np.int64, np.float64, covariant=True)
SType = TypeVar("SType", np.timedelta64, np.int64, np.float64, covariant=True)
RType = TypeVar("RType", np.int64, np.float64, covariant=True)

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
    dtype: str = 'object'

    def __init__(self, name: str, num_rows: Optional[int] = None):
        super().__init__(name=name, num_rows=num_rows)

    def __repr__(self) -> str:
        return f'<Generic[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'


class Nominal(ValueMeta[NType], Generic[NType]):
    """
    Nominal meta.

    Nominal describes any data that can be categorised, but has
    no quantitative interpretation, i.e it can only be given a name.
    Nominal data cannot be ordered nor compared, and there is no notion
    of 'closeness', e.g blood types ['AA', 'B', 'AB', 'O'].

    Attributes:
        categories: Optional; list of category names.
    """

    def __init__(
            self, name: str, categories: Optional[Sequence[NType]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name, num_rows=num_rows)
        self.categories: Optional[Sequence[NType]] = categories
        self.nan_freq = nan_freq

    def extract(self: NominalType, df: pd.DataFrame) -> NominalType:
        """Extract the categories and their relative frequencies from a data frame, if not already set."""
        super().extract(df)
        # Consider inf/-inf as nan
        df = df.copy().replace([np.inf, -np.inf], np.nan)
        if self.categories is None:
            self.categories = [c for c in np.array(df[self.name].dropna().unique(), dtype=self.dtype)]

        if self.nan_freq is None:
            self.nan_freq = df[self.name].isna().sum() / len(df) if len(df) > 0 else 0

        return self

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "nan_freq": self.nan_freq,
            "categories": [c for c in self.categories] if self.categories is not None else None
        })

        return d

    def __repr__(self) -> str:
        return f'<Nominal[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'


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
            self, name: str, categories: Optional[Sequence[OType]] = None, nan_freq: Optional[float] = None,
            min: Optional[OType] = None, max: Optional[OType] = None, num_rows: Optional[int] = None

    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)  # type: ignore
        self._min: Optional[OType] = min
        self._max: Optional[OType] = max

    def extract(self: OrdinalType, df: pd.DataFrame) -> OrdinalType:
        df = df.copy().replace('', np.nan)
        super().extract(df)
        assert self.categories is not None
        self.categories = self.sort(self.categories)

        if len(self.categories) > 0:
            self._min, self._max = self.categories[0], self.categories[-1]
        else:
            self._min, self._max = None, None

        return self

    @property
    def min(self) -> Optional[OType]:
        return self._min

    @property
    def max(self) -> Optional[OType]:
        return self._max

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

    def __repr__(self) -> str:
        return f'<Ordinal[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "_max": self.max,
            "_min": self.min
        })

        return d


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
            self, name: str, categories: Optional[Sequence[AType]] = None, nan_freq: Optional[float] = None,
            min: Optional[AType] = None, max: Optional[AType] = None, num_rows: Optional[int] = None,
            unit_meta: Optional['Scale'] = None
    ):
        super().__init__(
            name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows, min=min, max=max)
        self._unit_meta: Scale = Scale(name=f'{name}_unit') if unit_meta is None else unit_meta

    def extract(self: AffineType, df: pd.DataFrame) -> AffineType:
        super().extract(df)
        return self

    @property
    def unit_meta(self: AffineType) -> 'Scale[Any]':
        return self._unit_meta

    def __repr__(self) -> str:
        return f'<Affine[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'

    def sort(self, sr: Sequence[AType]) -> Sequence[AType]:
        return list(np.sort(sr))


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
            min: Optional[SType] = None, max: Optional[SType] = None, num_rows: Optional[int] = None,
            unit_meta: Optional['Scale'] = None
    ):
        super().__init__(
            name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows, min=min, max=max,
            unit_meta=unit_meta if unit_meta is not None else self
        )

    def extract(self: ScaleType, df: pd.DataFrame) -> ScaleType:
        super().extract(df)

        return self

    def __repr__(self) -> str:
        return f'<Scale[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'


class Ring(Scale[RType], Generic[RType]):

    def __init__(
            self, name: str, categories: Optional[Sequence[RType]] = None, nan_freq: Optional[float] = None,
            min: Optional[RType] = None, max: Optional[RType] = None, num_rows: Optional[int] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows, min=min, max=max)

    def extract(self: RingType, df: pd.DataFrame) -> RingType:
        super().extract(df)
        return self

    def __repr__(self) -> str:
        return f'<Ring[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'
