from abc import abstractmethod
from typing import Any, Generic, TypeVar, Optional, List, cast
from functools import cmp_to_key

import numpy as np
import pandas as pd

from .base_meta import Meta
from .exceptions import MetaNotExtractedError

DType = TypeVar('DType', covariant=True)
NType = TypeVar("NType", np.character, np.datetime64, np.integer, np.timedelta64, np.bool8, np.float64, covariant=True)
OType = TypeVar("OType", np.datetime64, np.integer, np.timedelta64, np.bool8, np.floating, covariant=True)
AType = TypeVar("AType", np.datetime64, np.integer, np.timedelta64, np.bool8, np.floating, covariant=True)
SType = TypeVar("SType", np.integer, np.timedelta64, np.bool8, np.floating, covariant=True)
RType = TypeVar("RType", np.bool8, np.floating, covariant=True)

ValueMetaType = TypeVar('ValueMetaType', bound='ValueMeta[Any]')
NominalType = TypeVar('NominalType', bound='Nominal[Any]')
OrdinalType = TypeVar('OrdinalType', bound='Ordinal[Any]')
AffineType = TypeVar('AffineType', bound='Affine[Any]')
ScaleType = TypeVar('ScaleType', bound='Scale[Any]')
RingType = TypeVar('RingType', bound='Ring[Any]')

KT = TypeVar('KT')
VT = TypeVar('VT', covariant=True)


class Domain(Generic[KT]):
    @abstractmethod
    def __contains__(self, item: KT) -> bool:
        pass

    @abstractmethod
    def tolist(self) -> List[KT]:
        pass


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
        return f'{self.__class__.__name__}(name={self.name}, dtype={self.dtype})'

    def __str__(self) -> str:
        return repr(self)

    def extract(self: ValueMetaType, df: pd.DataFrame) -> ValueMetaType:
        super().extract(df)
        self._extracted = True
        return self


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

    def __init__(self, name: str, domain: Optional[Domain[NType]] = None, nan_freq: Optional[float] = None):
        super().__init__(name=name)
        self.domain: Optional[Domain[NType]] = domain
        self.nan_freq = nan_freq

    def extract(self: NominalType, df: pd.DataFrame) -> NominalType:
        """Extract the domain and their relative frequencies from a data frame, if not already set."""
        super().extract(df)

        if self.domain is None:
            self.domain = self.infer_domain(df)

        return self

    def infer_domain(self, df: pd.DataFrame) -> Domain[NType]:
        return cast(Domain[NType], df[self.name].unique().tolist())


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
            self, name: str, domain: Optional[Domain[OType]] = None, nan_freq: Optional[float] = None,
            min: Optional[OType] = None, max: Optional[OType] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq)
        self.min: Optional[OType] = min
        self.max: Optional[OType] = max

    def extract(self: OrdinalType, df: pd.DataFrame) -> OrdinalType:
        super().extract(df)
        self.domain = cast(Domain[OType], self.domain)

        unique_sorted = self.sort(pd.Series(df[self.name].unique())).tolist()

        if self.min is None:
            self.min = unique_sorted[0]
        if self.max is None:
            self.max = unique_sorted[-1]

        return self

    def less_than(self, x: OType, y: OType) -> bool:
        """Return True if x < y"""
        if not self._extracted:
            raise MetaNotExtractedError
        self.domain = cast(Domain[OType], self.domain)

        if x not in self.domain or y not in self.domain:
            raise ValueError(f"x={x} or y={y} are not valid categories.")

        try:
            b = x < y
            if type(b) is not bool:
                raise TypeError
            return cast(bool, b)

        except TypeError:
            raise ValueError(f"x={x} or y={y} are not valid categories.")

    def _predicate(self, x: Any, y: Any) -> int:
        if self.less_than(x, y):
            return 1
        elif x == y:
            return 0
        else:
            return -1

    def sort(self, sr: pd.Series) -> pd.Series:
        """Sort pd.Series according to the ordering of this meta"""
        key = cmp_to_key(self._predicate)
        return pd.Series(sorted(sr, key=key, reverse=True))


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
    unit: Optional['Scale[Any]'] = None

    def __init__(
            self, name: str, domain: Optional[Domain[AType]] = None, nan_freq: Optional[float] = None,
            min: Optional[OType] = None, max: Optional[OType] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)

    def extract(self: AffineType, df: pd.DataFrame) -> AffineType:
        super().extract(df)
        return self


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
    unit: Optional['Scale[SType]'] = None

    def __init__(
            self, name: str, domain: Optional[Domain[SType]] = None, nan_freq: Optional[float] = None,
            min: Optional[OType] = None, max: Optional[OType] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)

    def extract(self: ScaleType, df: pd.DataFrame) -> ScaleType:
        super().extract(df)

        return self


class Ring(Scale[RType], Generic[RType]):
    class_name: str = 'Ring'

    def __init__(
            self, name: str, domain: Optional[Domain[RType]] = None, nan_freq: Optional[float] = None,
            min: Optional[OType] = None, max: Optional[OType] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)

    def extract(self: RingType, df: pd.DataFrame) -> RingType:
        super().extract(df)
        return self
