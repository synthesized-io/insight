from typing import List, Dict, Optional, Any, Type, TypeVar, MutableMapping, Iterator, Generic
from dataclasses import dataclass, field
from functools import cmp_to_key
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats

from .dtype import DType, NType, OType, AType, SType, RType
from .exceptions import UnknownDateFormatError

MetaType = TypeVar('MetaType', bound='Meta')
ValueMetaType = TypeVar('ValueMetaType', bound='ValueMeta')
NominalType = TypeVar('NominalType', bound='Nominal')
CategoricalType = TypeVar('CategoricalType', bound='Categorical')
OrdinalType = TypeVar('OrdinalType', bound='Ordinal')
AffineType = TypeVar('AffineType', bound='Affine')
ScaleType = TypeVar('ScaleType', bound='Scale')
RingType = TypeVar('RingType', bound='Ring')
DateType = TypeVar('DateType', bound='Date')


@dataclass(repr=False)
class Meta(MutableMapping[str, 'Meta']):
    """
    Base class for meta information that describes a dataset.

    Implements a hierarchical tree structure to describe arbitrary nested
    relations between data. Instances of Meta act as root nodes in the tree,
    and each branch can lead to another Meta or a leaf node, see ValueMeta.

    Attributes:
        name: a descriptive name.

    Examples:
        Custom nested structures can be easily built with this class, for example:

        >>> customer = Meta('customer')

        Add the leaf ValueMeta:

        >>> customer.age = Integer('age')
        >>> customer.title = Nominal('title')
        >>> customer.first_name = Nominal('first_name')

        Add address meta which acts as a root:

        >>> customer.address = Meta('customer_address')

        Add associated ValueMeta:

        >>> customer.address.street = Nominal('street')
        >>> customer.address.number = Nominal('number')

        >>> customer.bank = Meta('bank')
        >>> customer.bank.account = Nominal('account')
        >>> customer.bank.sort_code = Nominal('sort_code')

        Pretty print the tree structure:

        >>> print(customer)

        Meta objects are iterable, and allow iterating through the
        children:

        >>> for child_meta in customer:
        >>>     print(child_meta.name)
    """

    name: str
    _children: Dict[str, 'Meta'] = field(default_factory=dict)
    _extracted = False

    @property
    def children(self) -> List['Meta']:
        """Return the children of this Meta."""
        return [child for child in self.values()]

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'

    def extract(self: MetaType, x: pd.DataFrame) -> MetaType:
        """Extract the children of this Meta."""
        for child in self.children:
            child.extract(x)
        self._extracted = True
        return self

    def __setattr__(self, name: str, value):
        if isinstance(value, Meta) and not isinstance(self, ValueMeta):
            self[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        return self.get(name, object.__getattribute__(self, name))

    def __getitem__(self, k: str) -> 'Meta':
        return self._children[k]

    def __setitem__(self, k: str, v: 'Meta') -> None:
        if not isinstance(self, ValueMeta):
            self._children[k] = v

    def __delitem__(self, k: str) -> None:
        del self._children[k]

    def __iter__(self) -> Iterator[str]:
        for key in self._children:
            yield key

    def __len__(self) -> int:
        return len(self._children)

    def _tree_to_dict(self, meta: 'Meta') -> dict:
        """Convert nested Meta to dictionary"""
        if isinstance(meta, ValueMeta):
            d = {}
            for key, value in meta.__dict__.items():
                if not isinstance(value, Meta) and not key.startswith('_'):
                    d[key] = value
            return d
        elif isinstance(meta, Meta):
            return {n: m.to_dict() for n, m in meta.items()}

    def to_dict(self):

        d = {self.__class__.__name__: self._tree_to_dict(self)}
        for key, value in self.__dict__.items():
            if not isinstance(value, Meta) and not key.startswith('_'):
                d[self.__class__.__name__][key] = value
        return d

    @classmethod
    def from_dict(cls: Type['MetaType'], d: dict) -> MetaType:
        meta = cls(**d)
        if not issubclass(cls, ValueMeta):
            for attr_name, attrs in d.items():
                setattr(meta, attr_name, meta.from_dict(attrs))

        return meta


@dataclass(repr=False)
class DataFrameMeta(Meta):
    """
    Meta to describe an abitrary data frame.

    Each column is described by a derived ValueMeta object.

    Attributes:
        id_index: NotImplemented
        time_index: NotImplemented
        column_aliases: dictionary mapping column names to an alias.
    """
    id_index: Optional[str] = None
    time_index: Optional[str] = None
    column_aliases: Optional[Dict[str, str]] = None

    @property
    def column_meta(self) -> dict:
        """Get column <-> ValueMeta mapping."""
        col_meta = {}
        for child in self.children:
            col_meta[child.name] = child
        return col_meta


@dataclass(repr=False)
class ValueMeta(Meta, Generic[DType]):
    """
    Base class for meta information that describes a pandas series.

    ValueMeta objects act as leaf nodes in the Meta hierarchy. Derived
    classes must implement extract. All attributes should be set in extract.

    Attributes:
        name: The pd.Series name that this ValueMeta describes.
        dtype: Optional; The numpy dtype that this meta describes.
    """
    dtype: Optional[Type[DType]] = None

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, dtype={self.dtype})'

    def __str__(self):
        return repr(self)

    def extract(self: ValueMetaType, x: pd.DataFrame) -> ValueMetaType:
        super().extract(x)
        self.dtype = x[self.name].dtype
        self._extracted = True
        return self


@dataclass(repr=False)
class Nominal(ValueMeta[NType], Generic[NType]):
    """
    Nominal meta.

    Nominal describes any data that can be categorised, but has
    no quantitative interpretation, i.e it can only be given a name.
    Nominal data cannot be orderer nor compared, and there is no notion
    of 'closeness', e.g blood types ['AA', 'B', 'AB', 'O'].

    Attributes:
        categories: Optional; list of category names.
        probabilites: Optional; list of probabilites (relative frequencies) of each category.
    """
    categories: List[NType] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)

    def extract(self: NominalType, x: pd.DataFrame) -> NominalType:
        """Extract the categories and their relative frequencies from a data frame, if not already set."""
        super().extract(x)
        value_counts = x[self.name].value_counts(normalize=True, dropna=False, sort=False)
        if not self.categories:
            self.categories = value_counts.index.tolist()
            try:
                self.categories = sorted(self.categories)
            except TypeError:
                pass
        if not self.probabilities:
            self.probabilities = [value_counts[cat] if cat in value_counts else 0.0 for cat in self.categories]

        return self

    def probability(self, x: Any) -> float:
        """Get the probability mass of the category x."""
        try:
            return self.probabilities[self.categories.index(x)]
        except ValueError:
            if pd.isna(x):
                if not pd.isna(self.categories).any():
                    return 0.0
                try:
                    return self.probabilities[np.isnan(self.categories).argmax()]
                except TypeError:
                    return 0.0
            return 0.0

    @property
    def nan_freq(self) -> float:
        """Get NaN frequency."""
        return self.probability(np.nan)


@dataclass(repr=False)
class Constant(Nominal[NType]):
    """
    Constant meta.

    Constant describes data that has only a single value.

    Attributes:
        value: Optional; the constant value.
    """
    value: Optional[NType] = None

    def extract(self: 'Constant', x: pd.DataFrame) -> 'Constant':
        super().extract(x)
        self.value = x[self.name].dropna().iloc[0]
        return self


@dataclass(repr=False)
class Categorical(Nominal[NType], Generic[NType]):
    """
    Categorical meta.

    Categorical describes nominal data that can have a property of 'closeness', e.g
    a vocabulary where words with similar semantic meaning could be grouped together.

    Attributes:
        similarity_based:

    See also:
        Nominal
    """
    similarity_based: Optional[bool] = True


@dataclass(repr=False)
class Ordinal(Categorical[OType], Generic[OType]):
    """
    Ordinal meta.

    Ordinal meta describes categorical data that can be compared and ordered,
    and data can be arranged on a relative scale, e.g
    the Nandos scale: ['extra mild', 'mild', 'medium', 'hot', 'extra hot']

    Attributes:
        min: Optional; the minimum category
        max: Optional; the maximum category
    """
    min: Optional[OType] = None
    max: Optional[OType] = None

    def extract(self: OrdinalType, x: pd.DataFrame) -> OrdinalType:
        super().extract(x)
        if self.min is None:
            self.min = self.categories[0]
        if self.max is None:
            self.max = self.categories[-1]

        assert self.min is not None and self.max is not None and self.min < self.max

        return self

    def less_than(self, x: Any, y: Any) -> bool:
        """Return True if x < y"""
        if x not in self.categories or y not in self.categories:
            raise ValueError(f"x={x} or y={y} are not valid categories.")
        return self.categories.index(x) < self.categories.index(y)

    def _predicate(self, x: Any, y: Any) -> int:
        if self.less_than(x, y):
            return 1
        elif x == y:
            return 0
        else:
            return -1

    def sort(self, x: pd.Series) -> pd.Series:
        """Sort pd.Series according to the ordering of this meta"""
        key = cmp_to_key(self._predicate)
        return pd.Series(sorted(x, key=key, reverse=True))


@dataclass(repr=False)
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
        distribution: NotImplemented

        monotonic: Optional; True if data is monotonic, else False
    """
    distribution: Optional[scipy.stats.rv_continuous] = None
    monotonic: Optional[bool] = False

    def extract(self: AffineType, x: pd.DataFrame) -> AffineType:
        super().extract(x)
        if (np.diff(x[self.name]).astype(np.float64) > 0).all():
            self.monotonic = True
        else:
            self.monotonic = False

        return self


@dataclass(repr=False)
class Date(Affine[np.datetime64]):
    """
    Date meta.

    Date meta describes affine data than can be interpreted as a datetime,
    e.g the string '4/20/2020'.

    Attributes:
        date_format: Optional; string representation of date format, e.g '%d/%m/%Y'.
    """
    dtype = np.datetime64
    date_format: Optional[str] = None

    def extract(self: DateType, x: pd.DataFrame) -> DateType:
        super().extract(x)
        if self.date_format is None:
            self.date_format = get_date_format(x[self.name])

        x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)
        x[self.name] = x[self.name].dt.strftime(self.date_format)

        return self


@dataclass(repr=False)
class Scale(Affine[SType], Generic[SType]):
    """
    Scale meta.

    Scale describes data that can live in a vector space. Data points
    can be reached through linear combinations of other points,
    and points can also be multiplied by scalars.

    This describes pretty much most continuous types.

    Attributes:
        nonnegative: Optional; True if data is nonnegative, else False.
    """
    nonnegative: Optional[bool] = None

    def extract(self: ScaleType, x: pd.DataFrame) -> ScaleType:
        super().extract(x)
        if (x[self.name][~pd.isna(x[self.name])] >= 0).all():
            self.nonnegative = True
        else:
            self.nonnegative = False

        return self


@dataclass(repr=False)
class Integer(Scale[np.int64]):
    dtype = np.int64


@dataclass(repr=False)
class TimeDelta(Scale[np.timedelta64]):
    dtype = np.timedelta64


@dataclass(repr=False)
class Ring(Scale[RType], Generic[RType]):
    nonnegative: Optional[bool] = None

    def extract(self: RingType, x: pd.DataFrame) -> RingType:
        super().extract(x)
        if (x[self.name][~pd.isna(x[self.name])] >= 0).all():
            self.nonnegative = True
        else:
            self.nonnegative = False

        return self


@dataclass(repr=False)
class Bool(Ring[np.bool]):
    dtype = np.bool


@dataclass(repr=False)
class Float(Ring[np.float64]):
    dtype = np.float64


@dataclass(repr=False)
class AddressMeta(Meta):
    street: Optional[Nominal] = None
    number: Optional[Integer] = None
    postcode: Optional[Nominal] = None
    house_name: Optional[Nominal] = None
    flat: Optional[Nominal] = None
    city: Optional[Nominal] = None
    county: Optional[Nominal] = None


@dataclass(repr=False)
class BankMeta(Meta):
    account_number: Optional[Nominal] = None
    sort_code: Optional[Nominal] = None


@dataclass(repr=False)
class PersonMeta(Meta):
    title: Optional[Categorical] = None
    first_name: Optional[Nominal] = None
    last_name: Optional[Nominal] = None
    email: Optional[Nominal] = None
    age: Optional[Integer] = None
    gender: Optional[Categorical] = None
    mobile_telephone: Optional[Nominal] = None
    home_telephone: Optional[Nominal] = None
    work_telephone: Optional[Nominal] = None


def get_date_format(x: pd.Series) -> str:
    """
    Infer the date format for a series of dates.

    Returns:
        date format string, e.g "%d/%m/%Y.

    Raises:
        UnknownDateFormatError: date format cannot be inferred.
    """
    formats = (
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%d/%m/%Y %H.%M.%S', '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y', '%y-%m-%d', '%m-%d-%y', '%d-%m-%y',
        '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%y/%m/%d', '%m/%d/%y', '%d/%m/%y',
        '%y/%m/%d %H:%M', '%d/%m/%y %H:%M', '%m/%d/%y %H:%M'
    )
    parsed_format = None
    if x.dtype.kind == 'M':
        func = datetime.strftime
    else:
        func = datetime.strptime  # type: ignore
    for date_format in formats:
        try:
            x = x.apply(lambda x: func(x, date_format))
            parsed_format = date_format
        except ValueError:
            pass
        except TypeError:
            break

    if parsed_format is None:
        raise UnknownDateFormatError("Unable to infer date format.")
    return parsed_format


def func(x: Nominal) -> float:
    return x.probability(np.nan)

meta: Nominal[str] = Nominal('x')

def do_something_with_meta(x: Meta) -> Meta:
    return x

do_something_with_meta(meta)
