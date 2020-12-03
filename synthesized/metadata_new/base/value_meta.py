from abc import abstractmethod
from typing import Any, Generic, TypeVar, Optional, cast, Dict, Type, Union, Sequence
from functools import cmp_to_key

import numpy as np
import pandas as pd

from .domain import Domain, CategoricalDomain, OrderedDomain
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

    def __init__(self, name: str, domain: Optional[Domain[NType]] = None, nan_freq: Optional[float] = None):
        super().__init__(name=name)
        self._domain: Optional[Domain[NType]] = domain
        self.nan_freq = nan_freq

    def extract(self: NominalType, df: pd.DataFrame) -> NominalType:
        """Extract the domain and their relative frequencies from a data frame, if not already set."""
        super().extract(df)

        if self.domain is None:
            self.domain = self.infer_domain(df)

        if self.nan_freq is None:
            self.nan_freq = df[self.name].isna().sum() / len(df)

        return self

    @property
    def domain(self) -> Optional[Domain[NType]]:
        return self._domain

    @domain.setter
    def domain(self, x: Union[Dict[str, object], Optional[Domain[NType]]]):
        if type(x) is dict:
            x = cast(Dict[str, object], x)
            self._domain = Domain.STR_TO_DOMAIN[cast(str, x['domain_type'])].from_dict(x)
        else:
            x = cast(Optional[Domain[NType]], x)
            self._domain = x

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "nan_freq": self.nan_freq,
            "domain": self.domain if self.domain is None else self.domain.to_dict()
        })

        return d

    def infer_domain(self, df: pd.DataFrame) -> Domain[NType]:
        return CategoricalDomain(categories=df[self.name].unique().tolist())


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
        super().__init__(name=name, domain=domain, nan_freq=nan_freq)  # type: ignore
        self.min: Optional[OType] = min
        self.max: Optional[OType] = max

    def extract(self: OrdinalType, df: pd.DataFrame) -> OrdinalType:
        super().extract(df)
        self.domain = cast(Domain[OType], self.domain)

        categories = self.sort(df[self.name].unique())
        min_cat, max_cat = categories[0], categories[-1]
        self.min = min_cat
        self.max = max_cat

        return self

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "min": self.min,
            "max": self.max
        })

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

    def infer_domain(self, df: pd.DataFrame) -> Domain[OType]:
        categories = self.sort(df[self.name].unique())
        min_cat, max_cat = categories[0], categories[-1]
        return OrderedDomain(min=min_cat, max=max_cat)


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
            self, name: str, domain: Optional[Domain[AType]] = None, nan_freq: Optional[float] = None,
            min: Optional[OType] = None, max: Optional[OType] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)  # type: ignore

    def extract(self: AffineType, df: pd.DataFrame) -> AffineType:
        super().extract(df)
        return self

    @classmethod
    @abstractmethod
    def unit_meta(cls: Type[AffineType]) -> Type['Scale[Any]']:
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

    def __init__(
            self, name: str, domain: Optional[Domain[SType]] = None, nan_freq: Optional[float] = None,
            min: Optional[OType] = None, max: Optional[OType] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)  # type: ignore

    def extract(self: ScaleType, df: pd.DataFrame) -> ScaleType:
        super().extract(df)

        return self

    @classmethod
    def unit_meta(cls: Type[ScaleType]) -> Type[ScaleType]:
        return cls


class Ring(Scale[RType], Generic[RType]):
    class_name: str = 'Ring'

    def __init__(
            self, name: str, domain: Optional[Domain[RType]] = None, nan_freq: Optional[float] = None,
            min: Optional[OType] = None, max: Optional[OType] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)  # type: ignore

    def extract(self: RingType, df: pd.DataFrame) -> RingType:
        super().extract(df)
        return self
