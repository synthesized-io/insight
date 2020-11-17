from typing import List, Union, Dict, Optional, Any, TypeVar, Type
from dataclasses import dataclass, field
from functools import cmp_to_key
from datetime import datetime
import importlib
import functools

import numpy as np
import pandas as pd
import scipy.stats

from .exceptions import MetaNotExtractedError, UnsupportedDtypeError, UnknownDateFormatError

T = TypeVar('T', bound='Meta')
ValueMetaType = TypeVar('ValueMetaType', bound='ValueMeta')
NominalType = TypeVar('NominalType', bound='Nominal')
OrdinalType = TypeVar('OrdinalType', bound='Ordinal')
AffineType = TypeVar('AffineType', bound='Affine')


@dataclass
class MetaExtractorConfig():
    """
    Configuration parameters for MetaFactory.

    Attributes:
        categorical_threshold_log_multiplier: if number of unique values
        in a pd.Series is below this value a Categorical meta is returned.

        min_nim_unique: if number of unique values in pd.Series
        is below this a Categorical meta is returned.

        acceptable_nan_frac: when interpreting a series of type 'O',
        data is cast to numeric and non numeric types are cast to
        NaNs. If the frequency of NaNs is below this threshold, and
        Categorcial meta has not been inferred, then Float or Integer meta
        is returned.
    """
    categorical_threshold_log_multiplier: float = 2.5
    min_num_unique: int = 10
    acceptable_nan_frac: float = 0.25


@dataclass(repr=False)
class Meta():
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
    _children: List['Meta'] = field(default_factory=list)
    _extracted = False

    @property
    def children(self: T) -> List['Meta']:
        """Return the children of this Meta."""
        return self._children

    def __repr__(self: T):
        return f'{self.__class__.__name__}(name={self.name})'

    def __str__(self: T):
        return f"{_TreePrinter().print(self)}"

    def extract(self: T, x: pd.DataFrame) -> T:
        """Extract the children of this Meta."""
        for child in self._children:
            child.extract(x)
        self._extracted = True
        return self

    def register_child(self: T, child: 'Meta') -> None:
        if not isinstance(child, Meta):
            raise TypeError(f"cannot assign '{type(child)}' as child Meta '{child.name}'",
                            "(synthesized.metadata.meta.Meta required)")
        self._children.append(child)

    def __setattr__(self: T, name, value):
        if isinstance(value, Meta) and not isinstance(self, ValueMeta):
            if self._children is None:
                self._children = []
            self.register_child(value)
        object.__setattr__(self, name, value)

    def __getitem__(self: T, value):
        return getattr(self, value)

    def __iter__(self: T):
        for child in self._children:
            yield child

    def __len__(self: T):
        return len(self._children)

    def _tree_to_dict(self: T, meta: 'Meta') -> dict:
        """Convert nested Meta to dictionary"""
        if isinstance(meta, ValueMeta):
            d = {}
            for key, value in meta.__dict__.items():
                if not isinstance(value, Meta) and not key.startswith('_'):
                    d[key] = value
            return d
        elif isinstance(meta, Meta):
            return {m.name: m.to_dict() for m in meta}

    def to_dict(self: T):
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
            >>> customer.title = Nominal('title')
            >>> customer.address = Meta('customer_address')
            >>> customer.address.street = Nominal('street')

            Convert to dictionary:

            >>> customer.to_dict()
            {
                'Meta': {
                    'age': {
                        'Integer': {
                            'name': 'age',
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
                        'Meta': {
                            'street': {
                                'Nominal': {
                                    'name': 'street',
                                    'dtype': None,
                                    'categories': [],
                                    'probabilities': []
                                }
                            }
                            'name': 'customer_address'
                        }
                    }
                    'name': 'customer'
                }
            }


        See also:
            Meta.from_dict: construct a Meta from a dictionary
        """
        d = {self.__class__.__name__: self._tree_to_dict(self)}
        for key, value in self.__dict__.items():
            if not isinstance(value, Meta) and not key.startswith('_'):
                d[self.__class__.__name__][key] = value
        return d

    @classmethod
    def from_dict(cls: Type[T], d: dict) -> T:
        """
        Construct a Meta from a dictionary.

        See example in Meta.to_dict() for the required structure.

        See also:
            Meta.to_dict: convert a Meta to a dictionary
        """
        module = importlib.import_module("synthesized.metadata.meta")
        if isinstance(d, dict):
            name = list(d.keys())[0]
            if name in module.__dict__:
                meta = getattr(module, name)(name)
                if isinstance(meta, ValueMeta):
                    meta = getattr(module, name)(**d[name])
                else:
                    for attr_name, attrs in d[name].items():
                        setattr(meta, attr_name, meta.from_dict(attrs))
                return meta
            else:
                raise ValueError(f"class '{name}' is not a valid Meta in {module}.")

        else:
            return d


def requires_extract(func):
    """
    ValueMeta function decorator.

    Decorated bound methods of a ValueMeta raise a
    MetaNotExtractedError if ValueMeta.extract() has not
    been called.
    """
    @functools.wraps(func)
    def wrapper(cls, *args, **kwargs):
        func_name = func.__name__
        class_name = cls.__class__.__name__
        if not cls._extracted:
            raise MetaNotExtractedError(f"Must call {class_name}.extract() before {class_name}.{func_name}()")
        else:
            return func(cls, *args, **kwargs)
    return wrapper


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
        """
        Get column <-> ValueMeta mapping
        """
        col_meta = {}
        for child in self.children:
            col_meta[child.name] = child
        return col_meta


@dataclass(repr=False)
class ValueMeta(Meta):
    """
    Base class for meta information that describes a pandas series.

    ValueMeta objects act as leaf nodes in the Meta hierarchy. Derived
    classes must implement extract. All attributes should be set in extract.

    Attributes:
        name: The pd.Series name that this ValueMeta describes.
        dtype: Optional; The numpy dtype that this meta describes.
    """
    dtype: Optional[str] = None

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, dtype={self.dtype})'

    def __str__(self):
        return repr(self)

    def extract(self: ValueMetaType, x: pd.DataFrame) -> ValueMetaType:
        self.dtype = x[self.name].dtype
        self._extracted = True
        return self


@dataclass(repr=False)
class Nominal(ValueMeta):
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
    categories: List[str] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)

    def extract(self: NominalType, x: pd.DataFrame) -> NominalType:
        """Extract the categories and their relative frequencies from a data frame, if not already set."""
        value_counts = x[self.name].value_counts(normalize=True, dropna=False, sort=False)
        if not self.categories:
            self.categories = value_counts.index.tolist()
            try:
                self.categories = sorted(self.categories)
            except TypeError:
                pass
        if not self.probabilities:
            self.probabilities = [value_counts[cat] if cat in value_counts else 0.0 for cat in self.categories]
        return super().extract(x)

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
class Constant(Nominal):
    """
    Constant meta.

    Constant describes data that has only a single value.

    Attributes:
        value: Optional; the constant value.
    """
    value: Optional[Any] = None

    def extract(self, x: pd.DataFrame) -> 'Constant':
        self.value = x[self.name].dropna().iloc[0]
        return super().extract(x)


@dataclass(repr=False)
class Categorical(Nominal):
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
class Ordinal(Categorical):
    """
    Ordinal meta.

    Ordinal meta describes categorical data that can be compared and ordered,
    and data can be arranged on a relative scale, e.g
    the Nandos scale: ['extra mild', 'mild', 'medium', 'hot', 'extra hot']

    Attributes:
        min: Optional; the minimum category
        max: Optional; the maximum category
    """
    min: Optional[Union[str, float, int, pd.Timestamp]] = None
    max: Optional[Union[str, float, int, pd.Timestamp]] = None

    def extract(self: OrdinalType, x: pd.DataFrame) -> OrdinalType:
        self = super().extract(x)
        if self.min is None:
            self.min = self.categories[0]
        if self.max is None:
            self.max = self.categories[-1]
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
class Affine(Ordinal):
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
        if (np.diff(x[self.name]).astype(np.float64) > 0).all():
            self.monotonic = True
        else:
            self.monotonic = False
        return super().extract(x)

    # def probability(self, x):
    #     return self.distribution.pdf(x)


@dataclass(repr=False)
class Date(Affine):
    """
    Date meta.

    Date meta describes affine data than can be interpreted as a datetime,
    e.g the string '4/20/2020'.

    Attributes:
        date_format: Optional; string representation of date format, e.g '%d/%m/%Y'.
    """
    dtype: Optional[str] = 'datetime64[ns]'
    date_format: Optional[str] = None

    def extract(self, x: pd.DataFrame) -> 'Date':
        if self.date_format is None:
            self.date_format = get_date_format(x[self.name])

        x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)
        self = super().extract(x)
        x[self.name] = x[self.name].dt.strftime(self.date_format)
        return self


@dataclass(repr=False)
class Scale(Affine):
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

    def extract(self, x: pd.DataFrame) -> 'Scale':
        if (x[self.name][~pd.isna(x[self.name])] >= 0).all():
            self.nonnegative = True
        else:
            self.nonnegative = False
        return super().extract(x)


@dataclass(repr=False)
class Integer(Scale):
    dtype: Optional[str] = 'int64'


@dataclass(repr=False)
class TimeDelta(Scale):
    dtype: Optional[str] = 'timedelta64[ns]'


@dataclass(repr=False)
class Bool(Scale):
    dtype: Optional[str] = 'boolean'


@dataclass(repr=False)
class Float(Scale):
    dtype: Optional[str] = 'float64'


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


class _TreePrinter():
    """Helper class to pretty print the tree structure of a Meta."""
    def __init__(self):
        self._depth = 0
        self._string = ''

    def _print(self, tree: Meta):
        if not isinstance(tree, Meta):
            raise TypeError(f"tree must be 'Meta' not {type(tree)}")
        for child in tree:
            self._string += f'{self._depth * "    "}' + repr(child) + '\n'
            if len(child):
                self._string += self._depth * "    " + '|\n' + self._depth * "    " + '----' + '\n'
                self._depth += 1
                self._print(child)

        self._depth -= 1

    def print(self, tree):
        """Print the tree."""
        self._print(tree)
        return self._string


def _default_categorical(func):
    """
    MetaBuilder function decorator.

    Modifies the behaviour of a MetaBuilder function to return either a
    Categorical or Constant meta regardless of the underlying dtype. The decorated
    function will return either:

    1. Constant, for data with one unique value
    2. Categorical, for data with:
        number of unique values <= max(_MetaBuilder.min_num_unique, _MetaBuilder.categorical_threshold_log_multiplier * np.log(len(x))
        and if there are no genuine floats in the data
    3. Meta, i.e the return type of the decorated function if the these conditions are not met
    """
    @functools.wraps(func)
    def wrapper(cls, x: pd.Series) -> Union['Constant', 'Categorical', 'Meta']:
        n_unique = x.nunique()
        if n_unique == 1:
            return Constant(x.name)
        elif n_unique <= max(cls.min_num_unique, cls.categorical_threshold_log_multiplier * np.log(len(x))) \
                and (not _MetaBuilder._contains_genuine_floats(x)):
            return Categorical(x.name, similarity_based=True if n_unique > 2 else False)
        else:
            return func(cls, x, **cls.kwargs)
    return wrapper


class _MetaBuilder():
    """
    A functor class used internally by MetaFactory.

    Implements methods that return a derived Meta instance for a given pd.Series.
    The underyling numpy dtype (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
    determines the method that is called, and therefore the Meta that is returned.
    """
    def __init__(self, min_num_unique: int, acceptable_nan_frac: float, categorical_threshold_log_multiplier: float, **kwargs):
        self._dtype_builders = {
            'i': self._IntBuilder,
            'u': self._IntBuilder,
            'M': self._DateBuilder,
            'm': self._TimeDeltaBuilder,
            'b': self._BoolBuilder,
            'f': self._FloatBuilder,
            'O': self._ObjectBuilder
        }

        self.min_num_unique = min_num_unique
        self.acceptable_nan_frac = acceptable_nan_frac
        self.categorical_threshold_log_multiplier = categorical_threshold_log_multiplier

        self.kwargs = kwargs

    def __call__(self, x: pd.Series) -> ValueMeta:
        return self._dtype_builders[x.dtype.kind](x, **self.kwargs)

    def _DateBuilder(self, x: pd.Series, **kwargs) -> Date:
        return Date(x.name, **kwargs)

    def _TimeDeltaBuilder(self, x: pd.Series, **kwargs) -> TimeDelta:
        return TimeDelta(x.name, **kwargs)

    def _BoolBuilder(self, x: pd.Series, **kwargs) -> Bool:
        return Bool(x.name, **kwargs)

    @_default_categorical
    def _IntBuilder(self, x: pd.Series, **kwargs) -> Integer:
        return Integer(x.name, **kwargs)

    @_default_categorical
    def _FloatBuilder(self, x: pd.Series, **kwargs) -> Union[Float, Integer]:

        # check if is integer (in case NaNs which cast to float64)
        # delegate to __IntegerBuilder
        if self._contains_genuine_floats(x):
            return Float(x.name, **kwargs)
        else:
            return self._IntBuilder(x, **kwargs)

    def _CategoricalBuilder(self, x: pd.Series, **kwargs) -> Union[Ordinal, Categorical]:
        if isinstance(x.dtype, pd.CategoricalDtype):
            categories = x.cat.categories.tolist()
            if x.cat.ordered:
                return Ordinal(x.name, categories=categories, **kwargs)
            else:
                return Categorical(x.name, categories=categories, **kwargs)

        else:
            return Categorical(x.name, **kwargs)

    def _ObjectBuilder(self, x: pd.Series, **kwargs) -> Union[Nominal, Date, Categorical, Float, Integer]:
        try:
            get_date_format(x)
            return self._DateBuilder(x, **kwargs)
        except (UnknownDateFormatError, ValueError, TypeError, OverflowError):

            n_unique = x.nunique()
            n_rows = len(x)

            x_numeric = pd.to_numeric(x, errors='coerce')
            num_nan = x_numeric.isna().sum()

            if isinstance(x.dtype, pd.CategoricalDtype):
                return self._CategoricalBuilder(x, similarity_based=True if n_unique > 2 else False)

            elif (n_unique <= np.sqrt(n_rows)
                    or n_unique <= max(self.min_num_unique, self.categorical_threshold_log_multiplier * np.log(len(x)))) \
                    and (not self._contains_genuine_floats(x_numeric)):
                return self._CategoricalBuilder(x, similarity_based=True if n_unique > 2 else False)

            elif num_nan / n_rows < self.acceptable_nan_frac:
                if self._contains_genuine_floats(x_numeric):
                    return self._FloatBuilder(x_numeric)
                else:
                    return self._IntBuilder(x_numeric)

            else:
                return Nominal(x.name)

    @staticmethod
    def _contains_genuine_floats(x: pd.Series) -> bool:
        return (~x.dropna().apply(_MetaBuilder._is_integer_float)).any()

    @staticmethod
    def _is_integer_float(x: Any) -> bool:
        """Returns True if x can be represented as an integer."""
        try:
            return float(x).is_integer()
        except (ValueError, TypeError):
            return False


class MetaFactory():
    """Factory class to create Meta instances from pd.Series and pd.DataFrame objects."""
    def __init__(self, config: Optional[MetaExtractorConfig] = None):

        if config is None:
            self.config = MetaExtractorConfig()
        else:
            self.config = config

        self._builder = _MetaBuilder(**vars(self.config))

    def __call__(self, x: Union[pd.Series, pd.DataFrame]) -> Union[ValueMeta, DataFrameMeta]:
        return self.create_meta(x)

    def create_meta(self, x: Union[pd.Series, pd.DataFrame], name: Optional[str] = 'df') -> Union[ValueMeta, DataFrameMeta]:
        """
        Instantiate a Meta object from a pandas series or data frame.

        The underlying numpy dtype kind (e.g 'i', 'M', 'f') is used to determine the dervied Meta object for a series.

        Args:
            x: a pandas series or data frame for which to create the Meta instance
            name: Optional; The name of the instantianted DataFrameMeta if x is a data frame

        Returns:
            A derived ValueMeta instance or DataFrameMeta instance if x is a pd.Series or pd.DataFrame, respectively.

        Raises:
            UnsupportedDtypeError: The data type of the pandas series is not supported.
            TypeError: An error occured during instantiation of a ValueMeta.
        """
        if isinstance(x, pd.DataFrame):
            return self._from_df(x, name)
        elif isinstance(x, pd.Series):
            return self._from_series(x)
        else:
            raise TypeError(f"Cannot create meta from {type(x)}")

    def _from_series(self, x: pd.Series) -> ValueMeta:
        if x.dtype.kind not in self._builder._dtype_builders:
            raise UnsupportedDtypeError(f"'{x.dtype}' is unsupported")
        return self._builder(x)

    def _from_df(self, x: pd.DataFrame, name: Optional[str] = 'df') -> DataFrameMeta:
        if name is None:
            raise ValueError("name must not be a string, not None")
        meta = DataFrameMeta(name)
        for col in x.columns:
            try:
                child = self._from_series(x[col])
                setattr(meta, child.name, child)
            except TypeError as e:
                print(f"Warning. Encountered error when interpreting ValueMeta for '{col}'", e)
        return meta

    @staticmethod
    def default_config() -> MetaExtractorConfig:
        return MetaExtractorConfig(
            categorical_threshold_log_multiplier=2.5,
            acceptable_nan_frac=0.25, min_num_unique=10)


class MetaExtractor(MetaFactory):
    """Extract the DataFrameMeta for a data frame"""
    def __init__(self, config: Optional[MetaExtractorConfig] = None):
        super().__init__(config)

    @staticmethod
    def extract(x: pd.DataFrame, config: Optional[MetaExtractorConfig] = None) -> DataFrameMeta:
        """
        Instantiate and extract the DataFrameMeta that describes a data frame.

        Args:
            x: the data frame to instantiate and extract DataFrameMeta.
            config: Optional; The configuration parameters to MetaFactory.

        Returns:
            A DataFrameMeta instance for which all child meta have been extracted.

        Raises:
            UnsupportedDtypeError: The data type of a column in the data frame pandas is not supported.
            TypeError: An error occured during instantiation of a ValueMeta.
        """

        factory = MetaExtractor(config)
        df_meta = factory._from_df(x).extract(x)
        return df_meta


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
