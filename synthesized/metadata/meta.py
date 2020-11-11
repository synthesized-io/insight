from typing import List, Union, Dict
from dataclasses import dataclass
from functools import cmp_to_key
from datetime import datetime
import importlib

import numpy as np
import pandas as pd
import scipy.stats

MetaExtractorConfig = {
    'categorical_threshold_log_multiplier': 2.5,
    'acceptable_nan_frac': 0.25,
    'min_num_unique': 10
}


class MetaBuilder():
    """MetaFactory functor."""
    def __init__(self, min_num_unique, acceptable_nan_frac, categorical_threshold_log_multiplier, **kwargs):
        self._dtype_builders = {
            'i': self._IntBuilder,
            'u': self._IntBuilder,
            'M': self._DateBuilder,
            'm': self._TimeDeltaBuilder,
            '?': self._BoolBuilder,
            'f': self._FloatBuilder,
            'O': self._ObjectBuilder
        }

        self.min_num_unique = min_num_unique
        self.acceptable_nan_frac = acceptable_nan_frac
        self.categorical_threshold_log_multiplier = categorical_threshold_log_multiplier
        self._meta = None

        self.kwargs = kwargs

    def __call__(self, x: pd.Series):
        num_unique = x.nunique()
        if num_unique == 1:
            return Constant(x.name)
        else:
            return self._dtype_builders[x.dtype.kind](x, **self.kwargs)

    def _DateBuilder(self, x: pd.Series, **kwargs) -> 'Meta':
        self._meta = Date(x.name, **kwargs)
        return self._meta

    def _TimeDeltaBuilder(self, x: pd.Series, **kwargs) -> 'Meta':
        self._meta = TimeDelta(x.name, **kwargs)
        return self._meta

    def _BoolBuilder(self, x: pd.Series, **kwargs) -> 'Meta':
        self._meta = Bool(x.name, **kwargs)
        return self._meta

    def _IntBuilder(self, x: pd.Series, **kwargs) -> 'Meta':

        if x.nunique() == 2:
            self._meta = Bool(x.name, **kwargs)
        else:
            self._meta = Integer(x.name, **kwargs)

        return self._meta

    def _FloatBuilder(self, x: pd.Series, **kwargs) -> 'Meta':
        self._meta = Float(x.name, **kwargs)

        # check if is integer (in case NaNs which cast to float64)
        # delegate to __IntegerBuilder
        is_integer = x[~x.isna()].apply(lambda x: x.is_integer()).all()
        if is_integer:
            return self._IntBuilder(x, **kwargs)

        return self._meta

    def _CategoricalBuilder(self, x: pd.Series, **kwargs) -> 'Meta':
        if isinstance(x.dtype, pd.CategoricalDtype):
            categories = x.categories.tolist()
            if x.ordered:
                self._meta = Ordinal(x.name, categories=categories, **kwargs)
            else:
                self._meta = Categorical(x.name, categories=categories, **kwargs)

        else:
            self._meta = Categorical(x.name, **kwargs)
        return self._meta

    def _ObjectBuilder(self, x: pd.Series, **kwargs) -> 'Meta':
        try:
            pd.to_datetime(x)
            self._meta = self._DateBuilder(x, **kwargs)
        except (ValueError, TypeError, OverflowError):

            n_unique = x.nunique()
            n_rows = len(x)

            x_numeric = pd.to_numeric(x, errors='coerce')
            num_nan = x_numeric.isna().sum()

            if isinstance(x.dtype, pd.CategoricalDtype):
                self._meta = self._CategoricalBuilder(x, similarity_based=True if n_unique > 2 else False)

            elif num_nan / n_rows < self.acceptable_nan_frac:
                self._meta = self._FloatBuilder(x)

            elif (n_unique <= np.sqrt(n_rows) or
                  n_unique <= max(self.min_num_unique, self.categorical_threshold_log_multiplier * np.log(len(x)))) \
                    and (not MetaBuilder._contains_genuine_floats(x)):
                self._meta = self._CategoricalBuilder(x, similarity_based=True if n_unique > 2 else False)

            else:
                self._meta = ValueMeta(x.name)

        return self._meta

    @staticmethod
    def _contains_genuine_floats(x: pd.Series) -> bool:
        return x.dropna().apply(MetaBuilder._is_integer_float).any()

    @staticmethod
    def _is_integer_float(x) -> bool:
        """Returns True if x can be represented as an integer."""
        if isinstance(x, float):
            return x.is_integer()
        else:
            return False


class MetaFactory():

    def __init__(self, config=None):

        if config is None:
            self.config = MetaExtractorConfig
        else:
            self.config = config

        self._builder = MetaBuilder(**self.config)

    def __call__(self, x: Union[pd.Series, pd.DataFrame]) -> 'Meta':
        return self.create_meta(x)

    def create_meta(self, x: pd.DataFrame, name: str = 'df') -> 'Meta':

        if isinstance(x, pd.DataFrame):
            meta = DataFrameMeta(name)
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

        return meta


class MetaExtractor(MetaFactory):
    def __init__(self, config=None):
        super().__init__(config)

    @classmethod
    def extract(cls, x: pd.DataFrame, config=None) -> 'DataFrameMeta':
        factory = cls(config)
        df_meta = factory.create_meta(x)
        df_meta.extract(x)
        return df_meta


@dataclass(repr=False)
class Meta():

    name: str
    _children: List['Meta'] = None

    def __post_init__(self):
        if self._children is None:
            self._children = []

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'

    def __str__(self):
        return f"{TreePrinter().print(self)}"

    def extract(self, x: pd.DataFrame) -> 'Meta':
        for child in self._children:
            child.extract(x)

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
    def from_dict(cls, d):
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


class DataFrameMeta(Meta):
    id_index: str = None
    time_index: str = None
    column_aliases: Dict[str, str] = None

    @property
    def column_meta(self) -> pd.Series:
        col_meta = {}
        for child in self.children:
            col_meta[child.name] = child
        return pd.Series(col_meta)


@dataclass(repr=False)
class ValueMeta(Meta):

    dtype: str = None

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, dtype={self.dtype})'

    def extract(self, x: pd.DataFrame) -> 'Meta':
        self.dtype = x[self.name].dtype
        return self


@dataclass(repr=False)
class Nominal(ValueMeta):

    categories: List[str] = None
    probabilities: List[float] = None

    def extract(self, x: pd.DataFrame) -> 'Meta':
        super().extract(x)
        if self.categories is None:
            value_counts = x[self.name].value_counts(normalize=True, dropna=False)
            self.categories = value_counts.index.tolist()
        if self.probabilities is None:
            self.probabilities = [value_counts[cat] if cat in value_counts else 0.0 for cat in self.categories]
        return self

    def probability(self, x: Union[str, float, int]) -> float:
        try:
            return self.probabilities[self.categories.index(x)]
        except ValueError:
            if np.isnan(x):
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

    value = None

    def extract(self, x: pd.DataFrame) -> 'Meta':
        super().extract(x)
        self.value = x[self.name].dropna().iloc[0]


@dataclass(repr=False)
class Categorical(Nominal):

    similarity_based: bool = True


@dataclass(repr=False)
class Ordinal(Categorical):

    min: Union[str, float, int] = None
    max: Union[str, float, int] = None

    def extract(self, x: pd.DataFrame) -> 'Meta':
        super().extract(x)
        self.categories = sorted(self.categories)
        if self.min is None:
            self.min = self.categories[0]
        if self.max is None:
            self.max = self.categories[-1]
        return self

    def less_than(self, x, y) -> bool:
        if x not in self.categories or y not in self.categories:
            raise ValueError
        return self.categories.index(x) < self.categories.index(y)

    def _predicate(self, x, y) -> int:
        if self.less_than(x, y):
            return 1
        elif x == y:
            return 0
        else:
            return -1

    def sort(self, x: pd.Series) -> pd.Series:
        key = cmp_to_key(self._predicate)
        return sorted(x, key=key)


@dataclass(repr=False)
class Affine(Ordinal):

    distribution: scipy.stats.rv_continuous = None
    monotonic: bool = False

    def extract(self, x: pd.DataFrame) -> 'Meta':
        super().extract(x)
        if (np.diff(x[self.name]).astype(np.float64) > 0).all():
            self.monotonic = True
        else:
            self.monotonic = False

    # def probability(self, x):
    #     return self.distribution.pdf(x)


@dataclass(repr=False)
class Date(Affine):

    dtype: str = 'datetime64[ns]'
    date_format: str = None

    def extract(self, x: pd.DataFrame) -> 'Meta':
        if self.date_format is None:
            self.date_format = get_date_format(x[self.name])

        x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)
        super().extract(x)
        x[self.name] = x[self.name].dt.strftime(self.date_format)


@dataclass(repr=False)
class Scale(Affine):

    nonnegative: bool = None

    def extract(self, x: pd.DataFrame) -> 'Meta':
        super().extract(x)
        if (x[self.name] >= 0).all():
            self.nonnegative = True
        else:
            self.nonnegative = False
        return self


@dataclass(repr=False)
class Integer(Scale):
    dtype: str = 'int64'


@dataclass(repr=False)
class TimeDelta(Scale):
    dtype: str = 'timedelta64[ns]'


@dataclass(repr=False)
class Bool(Scale):
    dtype: str = 'boolean'
    categories = (0, 1)


@dataclass(repr=False)
class Float(Scale):
    dtype: str = 'float64'


@dataclass(repr=False)
class AddressMeta(Meta):
    street: Nominal = None
    number: Integer = None
    postcode: Nominal = None
    name: Nominal = None
    flat: Nominal = None
    city: Nominal = None
    county: Nominal = None
    postcode: Nominal = None


@dataclass(repr=False)
class BankMeta(Meta):
    account_number: Nominal = None
    sort_code: Nominal = None


@dataclass(repr=False)
class PersonMeta(Meta):
    title: Categorical = None
    first_name: Nominal = None
    last_name: Nominal = None
    email: Nominal = None
    age: Integer = None
    gender: Categorical = None
    mobile_telephone: Nominal = None
    home_telephone: Nominal = None
    work_telephone: Nominal = None


class TreePrinter():
    """Pretty print the tree structure of a Meta."""
    def __init__(self):
        self.depth = 0
        self.string = ''

    def _print(self, tree: Meta):
        if not isinstance(tree, Meta):
            raise TypeError(f"tree must be 'Meta' not {type(tree)}")
        for child in tree:
            self.string += f'{self.depth * "    "}' + repr(child) + '\n'
            if len(child):
                self.string += self.depth * "    " + '|\n' + self.depth * "    " + '----' + '\n'
                self.depth += 1
                self._print(child)

        self.depth -= 1

    def print(self, tree):
        self._print(tree)
        return self.string


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

    return parsed_format
