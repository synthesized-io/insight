from abc import abstractmethod
from typing import List, Dict, Optional, Any, Type, TypeVar, MutableMapping, Iterator, Generic, cast, Union
from typing_extensions import Protocol
from functools import cmp_to_key
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats

from .exceptions import UnknownDateFormatError, MetaNotExtractedError

DType = TypeVar('DType', covariant=True)
NType = TypeVar("NType", np.character, np.datetime64, np.integer, np.timedelta64, np.bool8, np.float64, covariant=True)
CType = TypeVar("CType", np.character, np.datetime64, np.integer, np.timedelta64, np.bool8, np.float64, covariant=True)
OType = TypeVar("OType", np.datetime64, np.integer, np.timedelta64, np.bool8, np.floating, covariant=True)
AType = TypeVar("AType", np.datetime64, np.integer, np.timedelta64, np.bool8, np.floating, covariant=True)
SType = TypeVar("SType", np.integer, np.timedelta64, np.bool8, np.floating, covariant=True)
RType = TypeVar("RType", np.bool8, np.floating, covariant=True)

MetaType = TypeVar('MetaType', bound='Meta')
ValueMetaType = TypeVar('ValueMetaType', bound='ValueMeta[Any]')
NominalType = TypeVar('NominalType', bound='Nominal[Any]')
CategoricalType = TypeVar('CategoricalType', bound='Categorical[Any]')
OrdinalType = TypeVar('OrdinalType', bound='Ordinal[Any]')
AffineType = TypeVar('AffineType', bound='Affine[Any]')
ScaleType = TypeVar('ScaleType', bound='Scale[Any]')
RingType = TypeVar('RingType', bound='Ring[Any]')
DateType = TypeVar('DateType', bound='Date')

KT_contra = TypeVar('KT_contra', contravariant=True)
VT_co = TypeVar('VT_co', covariant=True)


class Container(Protocol[KT_contra]):
    @abstractmethod
    def __contains__(self, item: KT_contra) -> bool:
        pass


class Projection(Protocol[KT_contra, VT_co]):
    @abstractmethod
    def __contains__(self, item: KT_contra) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, item: KT_contra) -> VT_co:
        pass


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

    def to_dict(self) -> Dict[str, object]:
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
    def from_dict(cls: Type['MetaType'], d: Dict[str, object]) -> 'MetaType':
        """
        Construct a Meta from a dictionary.
        See example in Meta.to_dict() for the required structure.
        See also:
            Meta.to_dict: convert a Meta to a dictionary
        """
        name = cast(str, d["name"])
        d.pop("class")

        extracted = d.pop("extracted")
        children = cast(Dict[str, Dict[str, object]], d.pop("children")) if "children" in d else None

        meta = cls(name=name)
        for attr, value in d.items():
            setattr(meta, attr, value)

        setattr(meta, '_extracted', extracted)

        if children is not None:
            meta_children = []
            for child in children.values():
                class_name = cast(str, child['class'])
                meta_children.append(STR_TO_META[class_name].from_dict(child))

            meta.children = meta_children

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
            column_aliases: Optional[Dict[str, str]] = None, num_columns: Optional[int] = None
    ):
        super().__init__(name=name)
        self.id_index = id_index
        self.time_index = time_index
        self.column_aliases = column_aliases if column_aliases is not None else {}
        self.num_columns = num_columns

    def extract(self, df: pd.DataFrame) -> 'DataFrameMeta':
        super().extract(df)
        self.num_columns = len(df.columns)
        return self

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
    def __init__(self, name: str, dtype: Optional[str] = None):
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
            self, name: str, dtype: Optional[str] = None, categories: Optional[Container[NType]] = None,
            probabilities: Optional[Projection[NType, float]] = None
    ):
        super().__init__(name=name, dtype=dtype)
        self.categories: Optional[Container[NType]] = categories
        self.probabilities: Optional[Projection[NType, float]] = probabilities

    def extract(self: NominalType, df: pd.DataFrame) -> NominalType:
        """Extract the categories and their relative frequencies from a data frame, if not already set."""
        super().extract(df)

        if self.categories is None:
            self.categories = self.infer_categories(df)
        if self.probabilities is None:
            self.probabilities = self.infer_probabilities(df)

        return self

    def infer_categories(self, df: pd.DataFrame) -> Projection[NType, float]:
        return cast(Projection[NType, float], df[self.name].unique().tolist())

    def infer_probabilities(self, df: pd.DataFrame) -> Projection[NType, float]:
        value_counts = df[self.name].value_counts(normalize=True, dropna=False, sort=False)
        return {cat: value_counts[cat] for cat in value_counts}

    def probability(self, x: Union[NType, None]) -> float:
        """Get the probability mass of the category x."""
        if not self._extracted:
            raise MetaNotExtractedError

        if x is None:
            return self.nan_freq

        x = cast(NType, x)

        self.probabilities = cast(Projection[NType, float], self.probabilities)
        self.categories = cast(Container[NType], self.categories)

        if x in self.probabilities:
            prob: float = self.probabilities.__getitem__(x)
        else:
            prob = 0.0

        return prob

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
    def __init__(self, name: str, value: Optional[NType] = None, dtype: Optional[str] = None):
        super().__init__(name=name, dtype=dtype)
        self.value: Optional[NType] = value

    def extract(self: 'Constant[NType]', df: pd.DataFrame) -> 'Constant[NType]':
        super().extract(df)
        self.value = df[self.name].dropna().iloc[0]
        return self


class Categorical(Nominal[CType], Generic[CType]):
    """
    Categorical meta.

    Categorical describes nominal data that can have a property of 'closeness', e.g
    a vocabulary where words with similar semantic meaning could be grouped together.

    Attributes:

    See also:
        Nominal
    """

    def __init__(
            self, name: str, dtype: Optional[str] = None, categories: Optional[Container[CType]] = None,
            probabilities: Optional[Projection[CType, float]] = None
    ):
        super().__init__(name=name, dtype=dtype, categories=categories, probabilities=probabilities)


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

    def __init__(
            self, name: str, dtype: Optional[str] = None, categories: Optional[Container[OType]] = None,
            probabilities: Optional[Projection[OType, float]] = None, min: Optional[OType] = None,
            max: Optional[OType] = None
    ):
        super().__init__(name=name, dtype=dtype, categories=categories, probabilities=probabilities)
        self.min: Optional[OType] = min
        self.max: Optional[OType] = max

    def extract(self: OrdinalType, df: pd.DataFrame) -> OrdinalType:
        super().extract(df)
        self.categories = cast(Container[OType], self.categories)
        self.probabilities = cast(Projection[OType, float], self.probabilities)

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
        self.categories = cast(Container[OType], self.categories)

        if x not in self.categories or y not in self.categories:
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
        distribution: NotImplemented
        monotonic: Optional; True if data is monotonic, else False
    """

    def __init__(
            self, name: str, dtype: Optional[str] = None, categories: Optional[Container[AType]] = None,
            probabilities: Optional[Projection[AType, float]] = None, min: Optional[AType] = None,
            max: Optional[AType] = None, monotonic: Optional[bool] = None,
            distribution: Optional[scipy.stats.rv_continuous] = None
    ):
        super().__init__(
            name=name, dtype=dtype, categories=categories, probabilities=probabilities, min=min, max=max
        )
        self.monotonic = monotonic
        self.distribution = distribution

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
    def __init__(
            self, name: str, dtype: str = 'datetime64[ns]', categories: Optional[Container[np.datetime64]] = None,
            probabilities: Optional[Projection[np.datetime64, float]] = None, min: Optional[np.datetime64] = None,
            max: Optional[np.datetime64] = None, monotonic: Optional[bool] = None,
            distribution: Optional[scipy.stats.rv_continuous] = None, date_format: Optional[str] = None
    ):
        super().__init__(
            name=name, dtype=dtype, categories=categories, probabilities=probabilities, min=min, max=max,
            monotonic=monotonic, distribution=distribution
        )
        self.date_format = date_format

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
    def __init__(
            self, name: str, dtype: Optional[str] = None, categories: Optional[Container[SType]] = None,
            probabilities: Optional[Projection[SType, float]] = None, min: Optional[SType] = None,
            max: Optional[SType] = None, monotonic: Optional[bool] = None,
            distribution: Optional[scipy.stats.rv_continuous] = None, nonnegative: Optional[bool] = None
    ):
        super().__init__(
            name=name, dtype=dtype, categories=categories, probabilities=probabilities, min=min, max=max,
            monotonic=monotonic, distribution=distribution
        )
        self.nonnegative = nonnegative

    def extract(self: ScaleType, df: pd.DataFrame) -> ScaleType:
        super().extract(df)
        if (df[self.name][~pd.isna(df[self.name])] >= 0).all():
            self.nonnegative = True
        else:
            self.nonnegative = False

        return self


class Integer(Scale[np.int64]):

    def __init__(
            self, name: str, dtype: str = 'int64', categories: Optional[Container[np.int64]] = None,
            probabilities: Optional[Projection[np.int64, float]] = None, min: Optional[np.int64] = None,
            max: Optional[np.int64] = None, monotonic: Optional[bool] = None,
            distribution: Optional[scipy.stats.rv_continuous] = None, nonnegative: Optional[bool] = None
    ):
        super().__init__(
            name=name, dtype=dtype, categories=categories, probabilities=probabilities, min=min, max=max,
            monotonic=monotonic, distribution=distribution, nonnegative=nonnegative
        )


class TimeDelta(Scale[np.timedelta64]):

    def __init__(
            self, name: str, dtype: str = 'timedelta64[ns]', categories: Optional[Container[np.timedelta64]] = None,
            probabilities: Optional[Projection[np.timedelta64, float]] = None, min: Optional[np.timedelta64] = None,
            max: Optional[np.timedelta64] = None, monotonic: Optional[bool] = None,
            distribution: Optional[scipy.stats.rv_continuous] = None, nonnegative: Optional[bool] = None
    ):
        super().__init__(
            name=name, dtype=dtype, categories=categories, probabilities=probabilities, min=min, max=max,
            monotonic=monotonic, distribution=distribution, nonnegative=nonnegative
        )


class Ring(Scale[RType], Generic[RType]):

    def __init__(
            self, name: str, dtype: Optional[str] = None, categories: Optional[Container[RType]] = None,
            probabilities: Optional[Projection[RType, float]] = None, min: Optional[RType] = None,
            max: Optional[RType] = None, monotonic: Optional[bool] = None,
            distribution: Optional[scipy.stats.rv_continuous] = None, nonnegative: Optional[bool] = None
    ):
        super().__init__(
            name=name, dtype=dtype, categories=categories, probabilities=probabilities, min=min, max=max,
            monotonic=monotonic, distribution=distribution, nonnegative=nonnegative
        )

    def extract(self: RingType, df: pd.DataFrame) -> RingType:
        super().extract(df)
        return self


class Bool(Ring[np.bool]):

    def __init__(
            self, name: str, dtype: str = 'bool', categories: Optional[Container[np.bool]] = None,
            probabilities: Optional[Projection[np.bool, float]] = None, min: Optional[np.bool] = None,
            max: Optional[np.bool] = None, monotonic: Optional[bool] = None,
            distribution: Optional[scipy.stats.rv_continuous] = None, nonnegative: Optional[bool] = None
    ):
        super().__init__(
            name=name, dtype=dtype, categories=categories, probabilities=probabilities, min=min, max=max,
            monotonic=monotonic, distribution=distribution, nonnegative=nonnegative
        )


class Float(Ring[np.float64]):

    def __init__(
            self, name: str, dtype: str = 'float64', categories: Optional[Container[np.float64]] = None,
            probabilities: Optional[Projection[np.float64, float]] = None, min: Optional[np.float64] = None,
            max: Optional[np.float64] = None, monotonic: Optional[bool] = None,
            distribution: Optional[scipy.stats.rv_continuous] = None, nonnegative: Optional[bool] = None
    ):
        super().__init__(
            name=name, dtype=dtype, categories=categories, probabilities=probabilities, min=min, max=max,
            monotonic=monotonic, distribution=distribution, nonnegative=nonnegative
        )


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
