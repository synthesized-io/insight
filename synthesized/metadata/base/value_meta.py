from functools import cmp_to_key
from typing import Any, Dict, Generic, Optional, Sequence, Type, TypeVar, cast

import numpy as np
import pandas as pd

from .meta import Meta
from ..exceptions import MetaNotExtractedError

DType = TypeVar('DType', covariant=True)
NType = TypeVar("NType", str, np.datetime64, np.timedelta64, np.int64, np.float64, np.bool8, covariant=True)
OType = TypeVar("OType", str, np.datetime64, np.timedelta64, np.int64, np.float64, np.bool8, covariant=True)
AType = TypeVar("AType", np.datetime64, np.timedelta64, np.int64, np.float64, covariant=True)
SType = TypeVar("SType", np.timedelta64, np.int64, np.float64, covariant=True)
RType = TypeVar("RType", np.int64, np.float64, covariant=True)

ValueMetaType = TypeVar('ValueMetaType', bound='ValueMeta', covariant=True)
NominalType = TypeVar('NominalType', bound='Nominal', covariant=True)
OrdinalType = TypeVar('OrdinalType', bound='Ordinal', covariant=True)
AffineType = TypeVar('AffineType', bound='Affine', covariant=True)
ScaleType = TypeVar('ScaleType', bound='Scale', covariant=True)
RingType = TypeVar('RingType', bound='Ring', covariant=True)


class ValueMeta(Meta[ValueMetaType], Generic[DType, ValueMetaType]):
    """
    Base class for meta information that describes a pandas series.

    ValueMeta objects act as leaf nodes in the Meta hierarchy. Derived
    classes must implement extract. All attributes should be set in extract.

    Attributes:
        name: The pd.Series name that this ValueMeta describes.
        dtype: Optional; The numpy dtype that this meta describes.
    """
    dtype: str = 'object'

    def __init__(self, name: str, children: Optional[Sequence[ValueMetaType]] = None, num_rows: Optional[int] = None):
        super().__init__(name=name, children=children, num_rows=num_rows)

    def __repr__(self) -> str:
        return f'<Generic[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'


class Nominal(ValueMeta[NType, ValueMetaType], Generic[NType, ValueMetaType]):
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
            self, name: str, children: Optional[Sequence[ValueMetaType]] = None,
            categories: Optional[Sequence[NType]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name, children=children, num_rows=num_rows)
        self._categories: Optional[Sequence[NType]] = categories
        self.nan_freq = nan_freq

    @property
    def categories(self) -> Sequence[NType]:
        if self._categories is None:
            raise MetaNotExtractedError(f"Meta '{self.name}' hasn't been extracted yet.")

        return [c for c in self._categories]

    @categories.setter
    def categories(self, c: Sequence[NType]) -> None:
        self._categories = c

    def extract(self: NominalType, df: pd.DataFrame) -> NominalType:
        """Extract the categories and their relative frequencies from a data frame, if not already set."""
        super().extract(df)
        # Consider inf/-inf as nan
        df = df.copy().replace([np.inf, -np.inf], np.nan)
        if self._categories is None:
            self.categories = [c for c in np.array(df[self.name].dropna().unique(), dtype=self.dtype)]

        if self.nan_freq is None:
            self.nan_freq = df[self.name].isna().sum() / len(df) if len(df) > 0 else 0

        return self

    def update_meta(self: NominalType, df: pd.DataFrame) -> NominalType:
        """Update the categories and nan_freq if required be"""

        cur_num_rows = self.num_rows
        super().update_meta(df)
        df = df.copy().replace([np.inf, -np.inf], np.nan)
        categories = [c for c in np.array(df[self.name].dropna().unique(), dtype=self.dtype)]

        if not set(categories).issubset(self.categories):
            self.categories = list(set([y for x in [categories, self.categories] for y in x]))

        num_nans = df[self.name].isna().sum()
        if cast(int, self.num_rows) > 0:
            if self.nan_freq is not None:
                self.nan_freq = (num_nans + self.nan_freq * cast(int, cur_num_rows)) / self.num_rows
            else:
                self.nan_freq = num_nans / self.num_rows

        return self

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "nan_freq": self.nan_freq,
            "categories": [c for c in self._categories] if self._categories is not None else None
        })

        return d

    @classmethod
    def from_dict(cls: Type[NominalType], d: Dict[str, object]) -> NominalType:

        name = cast(str, d["name"])
        extracted = cast(bool, d["extracted"])
        num_rows = cast(Optional[int], d["num_rows"])
        nan_freq = cast(Optional[float], d["nan_freq"])
        categories = cast(Optional[Sequence[NType]], d["categories"])
        children: Optional[Sequence[ValueMeta]] = ValueMeta.children_from_dict(d)

        meta = cls(name=name, children=children, num_rows=num_rows, nan_freq=nan_freq, categories=categories)
        meta._extracted = extracted

        return meta

    def __repr__(self) -> str:
        return f'<Nominal[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'


class Ordinal(Nominal[OType, ValueMeta], Generic[OType]):
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
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[OType]] = None, nan_freq: Optional[float] = None,
            min: Optional[OType] = None, max: Optional[OType] = None, num_rows: Optional[int] = None

    ):
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)  # type: ignore

    def extract(self: OrdinalType, df: pd.DataFrame) -> OrdinalType:
        df = df.copy().replace('', np.nan)
        super().extract(df)
        self.categories = self.sort(self.categories)

        return self

    def update_meta(self: OrdinalType, df: pd.DataFrame) -> OrdinalType:
        df = df.copy().replace('', np.nan)
        super().update_meta(df)
        self.categories = self.sort(self.categories)
        return self

    @property
    def min(self) -> Optional[OType]:
        return self.categories[0] if self.categories is not None and len(self.categories) > 0 else None

    @property
    def max(self) -> Optional[OType]:
        return self.categories[-1] if self.categories is not None and len(self.categories) > 0 else None

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
        if len(sr) == 0:
            return sr

        key = cmp_to_key(self._predicate)
        return sorted(sr, key=key, reverse=True)

    def __repr__(self) -> str:
        return f'<Ordinal[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'


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
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[AType]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None,
            unit_meta: Optional['Scale'] = None
    ):
        super().__init__(
            name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)
        self._unit_meta = unit_meta

    def extract(self: AffineType, df: pd.DataFrame) -> AffineType:
        if self._unit_meta is None:
            self._unit_meta = self._create_unit_meta()
            # self._unit_meta.categories is only populated here to identify an index column in model factory,
            # Index column won't have any missing values or NaNs
            if df[self.name].isna().sum() == 0:
                col_data = df[self.name].astype(self.dtype)
                self.unit_meta.categories = [
                    c for c in np.array(col_data.diff().dropna().unique(), dtype=self.unit_meta.dtype)
                ]

        super().extract(df)
        return self

    def update_meta(self: AffineType, df: pd.DataFrame) -> AffineType:
        if self._unit_meta is None:
            self._unit_meta = self._create_unit_meta()

        if df[self.name].isna().sum() == 0:
            col_data = df[self.name].astype(self.dtype)
            new_unit_meta_cats = [
                c for c in np.array(col_data.diff().dropna().unique(), dtype=self.unit_meta.dtype)
            ]

            try:
                cur_unit_meta_cats = self.unit_meta.categories
            except MetaNotExtractedError:
                cur_unit_meta_cats = []
            if not set(new_unit_meta_cats).issubset(cur_unit_meta_cats):
                self.unit_meta.categories = list(set([y for x in [cur_unit_meta_cats,
                                                 new_unit_meta_cats] for y in x]))

        super().update_meta(df)
        return self

    def _create_unit_meta(self) -> 'Scale[Any]':
        return Scale(f"{self.name}'", unit_meta=Scale(f"{self.name}''"))

    @property
    def unit_meta(self: AffineType) -> 'Scale[Any]':
        if self._unit_meta is None:
            raise MetaNotExtractedError
        return self._unit_meta

    def __repr__(self) -> str:
        return f'<Affine[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'

    def sort(self, sr: Sequence[AType]) -> Sequence[AType]:
        if len(sr) == 0:
            return sr
        return list(np.sort(sr))

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            'unit_meta': self._unit_meta.to_dict() if self._unit_meta is not None else None
        })
        return d

    @classmethod
    def from_dict(cls: Type[AffineType], d: Dict[str, object]) -> AffineType:

        name = cast(str, d["name"])
        extracted = cast(bool, d["extracted"])
        num_rows = cast(Optional[int], d["num_rows"])
        nan_freq = cast(Optional[float], d["nan_freq"])
        categories = cast(Optional[Sequence[AType]], d["categories"])
        children: Optional[Sequence[ValueMeta]] = ValueMeta.children_from_dict(d)
        d_unit = cast(Dict[str, object], d["unit_meta"]) if d["unit_meta"] is not None else None

        unit_meta = Scale.from_name_and_dict(cast(str, d_unit["class_name"]), d_unit) if d_unit is not None else None

        meta = cls(
            name=name, children=children, num_rows=num_rows, nan_freq=nan_freq, categories=categories,
            unit_meta=unit_meta
        )
        meta._extracted = extracted

        return meta


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
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[SType]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, unit_meta: Optional['Scale'] = None
    ):
        super().__init__(
            name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows,
            unit_meta=unit_meta
        )

    def extract(self: ScaleType, df: pd.DataFrame) -> ScaleType:
        super().extract(df)

        return self

    def update_meta(self: ScaleType, df: pd.DataFrame) -> ScaleType:
        super().update_meta(df)
        return self

    def _create_unit_meta(self: ScaleType) -> ScaleType:
        return type(self)(f"{self.name}'", unit_meta=type(self)(f"{self.name}''"))

    def __repr__(self) -> str:
        return f'<Scale[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'


class Ring(Scale[RType], Generic[RType]):

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[RType]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, unit_meta: Optional['Ring'] = None
    ):
        super().__init__(
            name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows,
            unit_meta=unit_meta
        )

    def extract(self: RingType, df: pd.DataFrame) -> RingType:
        super().extract(df)
        return self

    def update_meta(self: RingType, df: pd.DataFrame) -> RingType:
        super().update_meta(df)
        return self

    def __repr__(self) -> str:
        return f'<Ring[{self.dtype}]: {self.__class__.__name__}(name={self.name})>'
