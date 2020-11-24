# type: ignore
from typing import List, Dict, Optional, Any, Type, TypeVar, MutableMapping, Iterator, Generic, Union, cast
from functools import cmp_to_key
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats

from .exceptions import UnknownDateFormatError

DType = TypeVar('DType', covariant=True)
NType = TypeVar("NType", str, bytes, np.character, np.datetime64, np.integer, np.timedelta64, bool, np.bool8,
                np.float64, covariant=True)
CType = TypeVar("CType", str, bytes, np.character, np.datetime64, np.integer, np.timedelta64, bool, np.bool8,
                np.float64, covariant=True)
OType = TypeVar("OType", np.datetime64, np.integer, np.timedelta64, bool, np.bool8, np.floating, covariant=True)
AType = TypeVar("AType", np.datetime64, np.integer, np.timedelta64, bool, np.bool8, np.floating, covariant=True)
SType = TypeVar("SType", np.integer, np.timedelta64, bool, np.bool8, np.floating, covariant=True)
RType = TypeVar("RType", bool, np.bool8, np.floating, covariant=True)

MetaType = TypeVar('MetaType', bound='Meta')
ValueMetaType = TypeVar('ValueMetaType', bound='ValueMeta[Any]')
NominalType = TypeVar('NominalType', bound='Nominal[Any]')
CategoricalType = TypeVar('CategoricalType', bound='Categorical[Any]')
OrdinalType = TypeVar('OrdinalType', bound='Ordinal[Any]')
AffineType = TypeVar('AffineType', bound='Affine[Any]')
ScaleType = TypeVar('ScaleType', bound='Scale[Any]')
RingType = TypeVar('RingType', bound='Ring[Any]')
DateType = TypeVar('DateType', bound='Date')

JSON = Union[Dict[str, 'JSON'], List['JSON'], str, int, float, bool, None]


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

        >>> customer['age'] = Integer('age')
        >>> customer['title'] = Nominal[str]('title')
        >>> customer['first_name'] = Nominal[str]('first_name')

        Add address meta which acts as a root:

        >>> customer['address'] = Meta('customer_address')

        Add associated ValueMeta:

        >>> customer['address']['street'] = Nominal[str]('street')
        >>> customer['address']['number'] = Nominal[int]('number')

        >>> customer['bank'] = Meta('bank')
        >>> customer['bank']['account'] = Nominal[int]('account')
        >>> customer['bank']['sort_code'] = Nominal[int]('sort_code')


        Meta objects are iterable, and allow iterating through the
        children:

        >>> for child_meta in customer:
        >>>     print(child_meta)
    """
    def __init__(self, name: str):
        self.name = name
        self._children: Dict[str, 'Meta'] = dict()
        self._extracted: bool = False

    @property
    def children(self) -> List['Meta']:
        """Return the children of this Meta."""
        return [child for child in self.values()]

    @children.setter
    def children(self, children: List['Meta']) -> None:
        self._children = {child.name: child for child in children}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

    def extract(self, df: pd.DataFrame) -> 'Meta':
        """Extract the children of this Meta."""
        for child in self.children:
            child.extract(df)
        self._extracted = True
        return self

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

    def to_dict(self) -> Dict[str, JSON]:
        """
        Convert the Meta to a dictionary.
        The tree structure is converted to the following form:
        Meta.__class.__.__name__: {
            attr: value,
            value_meta_attr: {
                value_meta_attr.__class__.__name__: (**value_meta_attr.__dict__}
                )
            }
        }
        Examples:
            >>> customer = Meta('customer')
            >>> customer['title'] = Nominal('title')
            >>> customer['address'] = Meta('customer_address')
            >>> customer['address']['street'] = Nominal('street')
            Convert to dictionary:
            >>> customer.to_dict()
            {
                'name': 'customer',
                'class': 'Meta',
                'extracted': False,
                'children': {
                    'age': {
                        'name': 'age',
                        'class': 'Integer',
                        'extracted': False,
                        'dtype': 'int64',
                        'categories': [],
                        'probabilities': [],
                        'similarity_based': True,
                        'min': None,
                        'max': None,
                        'distribution': None,
                        'monotonic': False,
                        'nonnegative': None
                        }
                    },
                    'customer_address': {
                        'name': 'customer_address',
                        'class': 'Meta',
                        'extracted': False,
                        'children': {
                            'street': {
                                'name': 'street',
                                'class': 'Nominal',
                                'extracted': False,
                                'dtype': None,
                                'categories': [],
                                'probabilities': []
                            }
                        }
                    }
                }
            }
        See also:
            Meta.from_dict: construct a Meta from a dictionary
        """
        d = {
            "name": self.name,
            "class": self.__class__.__name__,
            "extracted": self._extracted
        }

        if len(self.children) > 0:
            d['children'] = {child.name: child.to_dict() for child in self.children}

        return d

    @classmethod
    def from_dict(cls: Type['MetaType'], d: Dict[str, JSON]) -> 'MetaType':
        """
        Construct a Meta from a dictionary.
        See example in Meta.to_dict() for the required structure.
        See also:
            Meta.to_dict: convert a Meta to a dictionary
        """
        name = d["name"]
        meta_class_name = d.pop("class")
        extracted = d.pop("extracted")
        children = d.pop("children") if "children" in d else None

        if type(meta_class_name) is not str:
            raise ValueError("Dict has invalid 'class' field.")

        meta_class_name = cast(str, meta_class_name)
        meta_class = STR_TO_META.get(meta_class_name)

        meta = meta_class(**d)
        if children is not None:
            meta.children = [Meta.from_dict(child) for child in children.values()]

        return meta


class DataFrameMeta(Meta):
    """
    Meta to describe an arbitrary data frame.

    Each column is described by a derived ValueMeta object.

    Attributes:
        id_index: NotImplemented
        time_index: NotImplemented
        column_aliases: dictionary mapping column names to an alias.
    """
    def __init__(
            self, name: str, id_index: Optional[str] = None, time_index: Optional[str] = None,
            column_aliases: Optional[Dict[str, str]] = None
    ):
        super().__init__(name=name)
        self.id_index = id_index
        self.time_index = time_index
        self.column_aliases = column_aliases if column_aliases is not None else {}

    @property
    def column_meta(self) -> Dict[str, Meta]:
        """Get column <-> ValueMeta mapping."""
        col_meta = {}
        for child in self.children:
            col_meta[child.name] = child
        return col_meta


class ValueMeta(Meta, Generic[DType]):
    """
    Base class for meta information that describes a pandas series.

    ValueMeta objects act as leaf nodes in the Meta hierarchy. Derived
    classes must implement extract. All attributes should be set in extract.

    Attributes:
        name: The pd.Series name that this ValueMeta describes.
        dtype: Optional; The numpy dtype that this meta describes.
    """
    def __init__(
            self, name: str, dtype: Optional[str] = None
    ):
        super().__init__(name=name)
        self.dtype = dtype

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, dtype={self.dtype})'

    def __str__(self) -> str:
        return repr(self)

    def extract(self: ValueMetaType, df: pd.DataFrame) -> ValueMetaType:
        super().extract(df)
        self.dtype = df[self.name].dtype.name
        self._extracted = True
        return self


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

    def __init__(
            self, name: str, dtype: Optional[str] = None, categories: Optional[List[NType]] = None,
            probabilities: Optional[List[float]] = None
    ):
        super().__init__(name=name, dtype=dtype)
        self.categories: Optional[List[NType]] = categories
        self.probabilities = probabilities

    def extract(self: NominalType, df: pd.DataFrame) -> NominalType:
        """Extract the categories and their relative frequencies from a data frame, if not already set."""
        super().extract(df)
        value_counts = df[self.name].value_counts(normalize=True, dropna=False, sort=False)
        if self.categories is None:
            self.categories = value_counts.index.tolist()
            try:
                self.categories = sorted(self.categories)
            except TypeError:
                pass
        if self.probabilities is None:
            self.probabilities = [value_counts[cat] if cat in value_counts else 0.0 for cat in self.categories]

        return self

    def probability(self, x: Any) -> float:
        """Get the probability mass of the category x."""
        if self.probabilities is None:
            raise
        try:
            return self.probabilities[self.categories.index(x)]
        except ValueError:
            if pd.isna(x):
                if not pd.isna(self.categories).any():
                    return 0.0
                try:
                    a: float = self.probabilities[np.isnan(self.categories).argmax()]
                    return a
                except TypeError:
                    return 0.0
            return 0.0

    @property
    def nan_freq(self) -> float:
        """Get NaN frequency."""
        return self.probability(np.nan)


class Constant(Nominal[NType]):
    """
    Constant meta.

    Constant describes data that has only a single value.

    Attributes:
        value: Optional; the constant value.
    """
    value: Optional[NType] = None

    def extract(self: 'Constant[NType]', df: pd.DataFrame) -> 'Constant[NType]':
        super().extract(df)
        self.value = df[self.name].dropna().iloc[0]
        return self


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

    def extract(self: OrdinalType, df: pd.DataFrame) -> OrdinalType:
        super().extract(df)
        if self.min is None:
            self.min = list(filter(pd.notnull, self.categories))[0]
        if self.max is None:
            self.max = list(filter(pd.notnull, self.categories))[-1]

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
        distribution: NotImplemented
        monotonic: Optional; True if data is monotonic, else False
    """
    distribution: Optional[scipy.stats.rv_continuous] = None
    monotonic: Optional[bool] = False

    def extract(self: AffineType, df: pd.DataFrame) -> AffineType:
        super().extract(df)
        if (np.diff(df[self.name]).astype(np.float64) > 0).all():
            self.monotonic = True
        else:
            self.monotonic = False

        return self


class Date(Affine[np.datetime64]):
    """
    Date meta.

    Date meta describes affine data than can be interpreted as a datetime,
    e.g the string '4/20/2020'.

    Attributes:
        date_format: Optional; string representation of date format, e.g '%d/%m/%Y'.
    """
    dtype: str = 'datetime64[ns]'
    date_format: Optional[str] = None

    def extract(self: DateType, df: pd.DataFrame) -> DateType:
        if self.date_format is None:
            try:
                self.date_format = get_date_format(df[self.name])
            except UnknownDateFormatError:
                self.date_format = None

        df[self.name] = pd.to_datetime(df[self.name], format=self.date_format)
        super().extract(df)  # call super here so we can get max, min from datetime.
        df[self.name] = df[self.name].dt.strftime(self.date_format)

        return self


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

    def extract(self: ScaleType, df: pd.DataFrame) -> ScaleType:
        super().extract(df)
        if (df[self.name][~pd.isna(df[self.name])] >= 0).all():
            self.nonnegative = True
        else:
            self.nonnegative = False

        return self


class Integer(Scale[np.int64]):
    dtype: str = 'int64'


class TimeDelta(Scale[np.timedelta64]):
    dtype: str = 'timedelta64'


class Ring(Scale[RType], Generic[RType]):
    nonnegative: Optional[bool] = None

    def extract(self: RingType, df: pd.DataFrame) -> RingType:
        super().extract(df)
        if (df[self.name][~pd.isna(df[self.name])] >= 0).all():
            self.nonnegative = True
        else:
            self.nonnegative = False

        return self


class Bool(Ring[np.bool]):
    dtype: str = 'bool'


class Float(Ring[np.float64]):
    dtype: str = 'float64'


def get_date_format(sr: pd.Series) -> str:
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
    if sr.dtype.kind == 'M':
        func = datetime.strftime
    else:
        func = datetime.strptime  # type: ignore
    for date_format in formats:
        try:
            sr = sr.apply(lambda x: func(x, date_format))
            parsed_format = date_format
        except ValueError:
            pass
        except TypeError:
            break

    if parsed_format is None:
        raise UnknownDateFormatError("Unable to infer date format.")
    return parsed_format


STR_TO_META: Dict[str, Type['Meta']] = {
    "Meta": Meta,
    "ValueMeta": ValueMeta,
    "Nominal": Nominal,
    "Categorical": Categorical,
    "Ordinal": Ordinal,
    "Affine": Affine,
    "Scale": Scale,
    "Ring": Ring,
    "Constant": Constant,
    "Date": Date,
    "Bool": Bool,
    "Integer": Integer,
    "TimeDelta": TimeDelta,
    "Float": Float
}
