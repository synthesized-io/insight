from typing import List, Union, Dict, Optional, Callable, Any
from dataclasses import dataclass
from functools import cmp_to_key
from datetime import datetime
import importlib
import functools

import numpy as np
import pandas as pd
import scipy.stats

MetaExtractorConfig = {
    'categorical_threshold_log_multiplier': 2.5,
    'acceptable_nan_frac': 0.25,
    'min_num_unique': 10
}


def _default_categorical(func):
    @functools.wraps(func)
    def wrapper(cls, x: pd.Series) -> Union['Constant', 'Categorical', 'Meta']:
        n_unique = x.nunique()
        if n_unique == 1:
            return Constant(x.name)
        elif n_unique <= max(cls.min_num_unique, cls.categorical_threshold_log_multiplier * np.log(len(x))) \
                and (not MetaBuilder._contains_genuine_floats(x)):
            return Categorical(x.name, similarity_based=True if n_unique > 2 else False)
        else:
            return func(cls, x, **cls.kwargs)
    return wrapper


class MetaBuilder():
    """MetaFactory functor."""
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

    def __call__(self, x: pd.Series) -> 'Meta':
        return self._dtype_builders[x.dtype.kind](x, **self.kwargs)

    def _DateBuilder(self, x: pd.Series, **kwargs) -> 'Date':
        return Date(x.name, **kwargs)

    def _TimeDeltaBuilder(self, x: pd.Series, **kwargs) -> 'TimeDelta':
        return TimeDelta(x.name, **kwargs)

    def _BoolBuilder(self, x: pd.Series, **kwargs) -> 'Bool':
        return Bool(x.name, **kwargs)

    @_default_categorical
    def _IntBuilder(self, x: pd.Series, **kwargs) -> 'Integer':
        return Integer(x.name, **kwargs)

    @_default_categorical
    def _FloatBuilder(self, x: pd.Series, **kwargs) -> Union['Float', 'Integer']:

        # check if is integer (in case NaNs which cast to float64)
        # delegate to __IntegerBuilder
        if self._contains_genuine_floats(x):
            meta = Float(x.name, **kwargs)
        else:
            meta = self._IntBuilder(x, **kwargs)
        return meta

    def _CategoricalBuilder(self, x: pd.Series, **kwargs) -> Union['Ordinal', 'Categorical']:
        if isinstance(x.dtype, pd.CategoricalDtype):
            categories = x.cat.categories.tolist()
            if x.cat.ordered:
                meta = Ordinal(x.name, categories=categories, **kwargs)  # type: Union[Ordinal, Categorical]
            else:
                meta = Categorical(x.name, categories=categories, **kwargs)

        else:
            meta = Categorical(x.name, **kwargs)
        return meta

    def _ObjectBuilder(self, x: pd.Series, **kwargs) -> Union['Nominal', 'Date', 'Categorical', 'Float', 'Integer']:
        try:
            get_date_format(x)
            meta = self._DateBuilder(x, **kwargs)  # type: Union[Nominal, Date, Categorical, Float, Integer]
        except (UnknownDateFormatError, ValueError, TypeError, OverflowError):

            n_unique = x.nunique()
            n_rows = len(x)

            x_numeric = pd.to_numeric(x, errors='coerce')
            num_nan = x_numeric.isna().sum()

            if isinstance(x.dtype, pd.CategoricalDtype):
                meta = self._CategoricalBuilder(x, similarity_based=True if n_unique > 2 else False)

            elif (n_unique <= np.sqrt(n_rows) or
                  n_unique <= max(self.min_num_unique, self.categorical_threshold_log_multiplier * np.log(len(x)))) \
                    and (not MetaBuilder._contains_genuine_floats(x_numeric)):
                meta = self._CategoricalBuilder(x, similarity_based=True if n_unique > 2 else False)

            elif num_nan / n_rows < self.acceptable_nan_frac:
                if MetaBuilder._contains_genuine_floats(x_numeric):
                    meta = self._FloatBuilder(x_numeric)
                else:
                    meta = self._IntBuilder(x_numeric)

            else:
                meta = Nominal(x.name)

        return meta

    @staticmethod
    def _contains_genuine_floats(x: pd.Series) -> bool:
        return (~x.dropna().apply(MetaBuilder._is_integer_float)).any()

    @staticmethod
    def _is_integer_float(x: Any) -> bool:
        """Returns True if x can be represented as an integer."""
        try:
            return float(x).is_integer()
        except (ValueError, TypeError):
            return False


class MetaFactory():

    def __init__(self, config: Optional[dict] = None):

        if config is None:
            self.config = MetaExtractorConfig
        else:
            self.config = config

        self._builder = MetaBuilder(**self.config)

    def __call__(self, x: Union[pd.Series, pd.DataFrame]) -> 'Meta':
        return self.create_meta(x)

    def create_meta(self, x: Union[pd.DataFrame, pd.DataFrame], name: Optional[str] = 'df') -> 'Meta':

        if name is None:
            # to please mypy...
            raise ValueError("name must not be None")

        if isinstance(x, pd.DataFrame):
            meta = DataFrameMeta(name)  # type: Union[DataFrameMeta, Meta]
            for col in x.columns:
                try:
                    child = self.create_meta(x[col], col)
                    setattr(meta, child.name, child)
                except TypeError as e:
                    print(f"Warning. Unable to interpret Meta for '{col}'", e)

        elif isinstance(x, pd.Series):
            if x.dtype.kind not in self._builder._dtype_builders:
                raise TypeError(f"'{x.dtype}' is unsupported")
            meta = self._builder(x)

        else:
            raise TypeError(f"Cannot create meta from {type(x)}")

        return meta


class MetaExtractor(MetaFactory):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

    @classmethod
    def extract(cls, x: pd.DataFrame, config: Optional[dict] = None) -> 'Meta':
        factory = cls(config)
        df_meta = factory.create_meta(x).extract(x)  # type: Meta
        return df_meta


@dataclass(repr=False)
class Meta():

    name: str
    _children: Optional[List['Meta']] = None
    _extracted = False

    def __post_init__(self):
        if self._children is None:
            self._children = []

    @property
    def children(self) -> List['Meta']:
        return self._children

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'

    def __str__(self):
        return f"{TreePrinter().print(self)}"

    def extract(self, x: pd.DataFrame) -> 'Meta':
        for child in self._children:
            child.extract(x)
        self._extracted = True
        return self

    def register_child(self, child: 'Meta') -> None:
        if not isinstance(child, Meta):
            raise TypeError(f"cannot assign '{type(child)}' as child Meta '{child.name}'",
                            "(synthesized.metadata.meta.Meta required)")
        self._children.append(child)

    def __setattr__(self, name, value):
        if isinstance(value, Meta) and not isinstance(self, ValueMeta):
            if self._children is None:
                self._children = []
            self.register_child(value)
        object.__setattr__(self, name, value)

    def __getitem__(self, value):
        return getattr(self, value)

    def __iter__(self):
        for child in self._children:
            yield child

    def __len__(self):
        return len(self._children)

    def _tree_to_dict(self, meta: 'Meta') -> dict:
        if isinstance(meta, ValueMeta):
            d = {}
            for key, value in meta.__dict__.items():
                if not isinstance(value, Meta) and not key.startswith('_'):
                    d[key] = value
            return d
        elif isinstance(meta, Meta):
            return {m.name: m.to_dict() for m in meta}

    def to_dict(self):
        d = {self.__class__.__name__: self._tree_to_dict(self)}
        for key, value in self.__dict__.items():
            if not isinstance(value, Meta) and not key.startswith('_'):
                d[self.__class__.__name__][key] = value
        return d

    @classmethod
    def from_dict(cls, d: dict):
        module = importlib.import_module("synthesized.metadata.meta")
        if isinstance(d, dict):
            name = list(d.keys())[0]
            if name in module.__dict__:
                cls = getattr(module, name)(name)
                if isinstance(cls, ValueMeta):
                    cls = getattr(module, name)(**d[name])
                else:
                    for attr_name, attrs in d[name].items():
                        setattr(cls, attr_name, cls.from_dict(attrs))
                return cls
            else:
                raise ValueError(f"class '{name}' is not a valid Meta in {module}.")

        else:
            return d


class MetaNotExtractedError(Exception):
    pass


def requires_extract(func):
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
    id_index: Optional[str] = None
    time_index: Optional[str] = None
    column_aliases: Optional[Dict[str, str]] = None

    @property
    def column_meta(self) -> pd.Series:
        col_meta = {}
        for child in self.children:
            col_meta[child.name] = child
        return pd.Series(col_meta)


@dataclass(repr=False)
class ValueMeta(Meta):

    dtype: Optional[str] = None

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, dtype={self.dtype})'

    def __str__(self):
        return repr(self)

    def extract(self, x: pd.DataFrame) -> 'ValueMeta':
        self.dtype = x[self.name].dtype
        self._extracted = True
        return self


@dataclass(repr=False)
class Nominal(ValueMeta):

    categories: Optional[List[str]] = None
    probabilities: Optional[List[float]] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if self.probabilities is None:
            self.probabilities = []

    def extract(self, x: pd.DataFrame) -> 'Nominal':
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
        return self.probability(np.nan)


@dataclass(repr=False)
class Constant(Nominal):

    value: Optional[Any] = None

    def extract(self, x: pd.DataFrame) -> 'Constant':
        self.value = x[self.name].dropna().iloc[0]
        return super().extract(x)


@dataclass(repr=False)
class Categorical(Nominal):

    similarity_based: Optional[bool] = True


@dataclass(repr=False)
class Ordinal(Categorical):

    min: Optional[Union[str, float, int]] = None
    max: Optional[Union[str, float, int]] = None

    def extract(self, x: pd.DataFrame) -> 'Ordinal':
        self = super().extract(x)
        if self.min is None:
            self.min = self.categories[0]
        if self.max is None:
            self.max = self.categories[-1]
        return self

    def less_than(self, x: Any, y: Any) -> bool:
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
        key = cmp_to_key(self._predicate)
        return pd.Series(sorted(x, key=key, reverse=True))


@dataclass(repr=False)
class Affine(Ordinal):

    distribution: Optional[scipy.stats.rv_continuous] = None
    monotonic: Optional[bool] = False

    def extract(self, x: pd.DataFrame) -> 'Affine':
        if (np.diff(x[self.name]).astype(np.float64) > 0).all():
            self.monotonic = True
        else:
            self.monotonic = False
        return super().extract(x)

    # def probability(self, x):
    #     return self.distribution.pdf(x)


@dataclass(repr=False)
class Date(Affine):

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


class TreePrinter():
    """Pretty print the tree structure of a Meta."""
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
        self._print(tree)
        return self._string


def get_date_format(x: pd.Series) -> pd.Series:
    """Infer date format."""
    formats = (
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%d/%m/%Y %H.%M.%S', '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y', '%y-%m-%d', '%m-%d-%y', '%d-%m-%y',
        '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%y/%m/%d', '%m/%d/%y', '%d/%m/%y',
        '%y/%m/%d %H:%M', '%d/%m/%y %H:%M', '%m/%d/%y %H:%M'
    )
    parsed_format = None
    for date_format in formats:
        try:
            x = x.apply(lambda x: datetime.strptime(x, date_format))
            parsed_format = date_format
        except ValueError:
            pass
        except TypeError:
            break

    if parsed_format is None:
        raise UnknownDateFormatError("Unable to infer date format.")
    return parsed_format


class UnknownDateFormatError(Exception):
    pass
