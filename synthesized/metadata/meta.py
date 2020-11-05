from typing import List, Union, Dict
from dataclasses import dataclass
from functools import cmp_to_key
from datetime import datetime

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

    @classmethod
    def create_meta(cls, x: pd.DataFrame, name: str = 'df', config=None) -> 'Meta':
        obj = cls(config)
        return obj._create_meta(x, name)

    def _create_meta(self, x: pd.DataFrame, name: str = 'df') -> 'Meta':

        if isinstance(x, pd.DataFrame):
            meta = DataFrameMeta(name)
            for col in x.columns:
                try:
                    child = self._create_meta(x[col], col)
                    meta.register_child(child)
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
        df_meta = factory._create_meta(x)
        df_meta.extract(x)
        return df_meta


@dataclass
class Meta():

    name: str
    children: List['Meta'] = None
    is_root = True

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def extract(self, x: pd.DataFrame) -> 'Meta':
        for child in self.children:
            child.extract(x)

    def register_child(self, child: 'Meta') -> None:
        self.children.append(child)

    def __setattr__(self, name, value):
        if isinstance(value, Meta) and self.is_root:
            if self.children is None:
                self.children = []
            self.register_child(value)
        object.__setattr__(self, name, value)

    def __getitem__(self, value):
        if self.children:
            for child in self.children:
                if child.name == value:
                    return child[value]
        elif value == self.name:
            return self
        else:
            raise KeyError(f"'{value}' is not a registered Meta.")

    def to_json(self):
        raise NotImplementedError

    def from_json(self):
        raise NotImplementedError


@dataclass(repr=False)
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


@dataclass
class ValueMeta(Meta):

    dtype: str = None
    is_root: bool = False


@dataclass
class Nominal(ValueMeta):

    categories: List[str] = None
    probabilities: List[float] = None

    def extract(self, x: pd.DataFrame) -> 'Meta':

        if self.categories is None:
            value_counts = x[self.name].value_counts(normalize=True, dropna=False)
            self.categories = value_counts.index.values.tolist()
        if self.probabilities is None:
            self.probabilities = [value_counts[cat] if cat in value_counts else 0.0 for cat in self.categories]
        return self

    def probability(self, x: Union[str, float, int]) -> float:
        try:
            return self.probabilities[self.categories.index(x)]
        except ValueError:
            if np.isnan(x):
                try:
                    return self.probabilities[np.isnan(self.categories).argmax()]
                except TypeError:
                    return 0.0
            return 0.0

    @property
    def nan_freq(self) -> float:
        return self.probability(np.nan)


@dataclass
class Constant(Nominal):

    value = None

    def extract(self, x: pd.DataFrame) -> 'Meta':
        super().extract(x)
        self.value = x[self.name].dropna().iloc[0]


@dataclass
class Categorical(Nominal):

    similarity_based: bool = True


@dataclass
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


@dataclass
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


@dataclass
class Date(Affine):

    dtype: str = 'datetime64[ns]'
    date_format: str = None

    def extract(self, x: pd.DataFrame) -> 'Meta':
        if self.date_format is None:
            self.date_format = get_date_format(x[self.name])

        x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)
        super().extract(x)
        x[self.name] = x[self.name].dt.strftime(self.date_format)


@dataclass
class Scale(Affine):

    nonnegative: bool = None

    def extract(self, x: pd.DataFrame) -> 'Meta':
        super().extract(x)
        if (x[self.name] >= 0).all():
            self.nonnegative = True
        else:
            self.nonnegative = False
        return self


@dataclass
class Integer(Scale):
    dtype: str = 'int64'


@dataclass
class TimeDelta(Scale):
    dtype: str = 'timedelta64[ns]'


@dataclass
class Bool(Scale):
    dtype: str = 'boolean'
    categories = (0, 1)


@dataclass
class Float(Scale):
    dtype: str = 'float64'


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
