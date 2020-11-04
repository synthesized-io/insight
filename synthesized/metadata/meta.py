from typing import List, Union, Tuple, Optional
from dataclasses import dataclass
from functools import cmp_to_key
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats

MetaExtractorConfig = {
    'categorical_threshold_log_multiplier': 2.5,
    'parsing_nan_fraction_threshold': 0.25,
    'min_num_unique': 10
}


class _mBuilder():

    def __init__(self):
        self._meta = None

    def __call__(self, x: pd.Series) -> 'Meta':
        self._meta = Date(x.name)
        return self._meta


class _iBuilder():

    def __init__(self):
        self._meta = None

    def __call__(self, x: pd.Series) -> 'Meta':

        if x.nunique == 2:
            self._meta = Bool(x.name)
        else:
            self._meta = Integer(x.name)

        return self._meta


class _fBuilder():

    def __init__(self):
        self._meta = None

    def __call__(self, x: pd.Series) -> 'Meta':
        self._meta = Float(x.name)

        return self._meta


class _oBuilder():

    def __init__(self):
        self._meta = None

    def __call__(self, x: pd.Series) -> 'Meta':
        try:
            pd.to_datetime(x)
            self._meta = Date(x.name)
        except (ValueError, TypeError, OverflowError):
            pass


class _cBuilder():

    def __init__(self):
        self._meta = None

    def __call__(self, x: pd.Series) -> 'Meta':
        if x.ordered:
            self._meta = Ordinal(x.name)
        else:
            self._meta = Categorical(x.name)

        return self._meta


class MetaFactory():

    def __init__(self, config=None):

        self._builders = {}
        if config is None:
            self.config = MetaExtractorConfig
        else:
            self.config = config

    def create_meta(self, x: pd.DataFrame, name: str = 'df') -> 'Meta':
        meta = Meta(name)
        for col in x.columns:
            child = self.identify_value(x[col], col)[0]
            meta.register_child(child)

        return meta

    def infer_meta(self, x: pd.Series, name: str = None) -> Tuple[Optional['Meta'], Optional[str]]:
        """Autodetect the meta describing a data frame column."""

        if name is None:
            name = x.name

        num_unique = x.nunique()
        if num_unique <= 1:
            return Constant(name)

        # ========== Non-numeric values ==========

        # Categorical value if small number of distinct values (or if data-set is too small)
        elif num_unique <= max(float(self.config['min_num_unique']),
                               self.config['categorical_threshold_log_multiplier'] * np.log(num_data)):
            # is_nan = df.isna().any()
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:
                    value = Categorical(name, similarity_based=True)
                    reason = "Small (< log(N)) number of distinct values. "
                else:
                    value = Categorical(name,)
                    reason = "Small (< log(N)) number of distinct values (= 2). "

        # Date value
        elif col.dtype.kind == 'M':  # 'm' timedelta
            value = Date(name)
            reason = "Column dtype kind is 'M'. "

        # Boolean value
        elif col.dtype.kind == 'b':
            # is_nan = df.isna().any()
            value = Bool(name)
            reason = "Column dtype kind is 'b'. "

        # Continuous value if integer (reduced variability makes similarity-categorical more likely)
        elif col.dtype.kind in ['i', 'u']:
            value = Integer(name)
            reason = f"Column dtype kind is '{col.dtype.kind}'. "

        # Categorical value if object type has attribute 'categories'
        elif col.dtype.kind == 'O' and hasattr(col.dtype, 'categories'):
            # is_nan = df.isna().any()
            if num_unique > 2:
                value = Categorical(name)
                reason = "Column dtype kind is 'O' and has 'categories' (> 2). "
            else:
                value = Categorical(name, similarity_based=False)
                reason = "Column dtype kind is 'O' and has 'categories' (= 2). "

        # Date value if object type can be parsed
        elif col.dtype.kind == 'O' and excl_nan_dtype.kind not in ['f', 'i']:
            try:
                date_data = pd.to_datetime(col)
                num_nan = date_data.isna().sum()
                if num_nan / num_data < self.config['parsing_nan_fraction_threshold']:
                    assert date_data.dtype.kind == 'M'
                    value = Date(name)
                    reason = "Column dtype is 'O' and convertable to datetime. "

            except (ValueError, TypeError, OverflowError):
                pass

        # Similarity-based categorical value if not too many distinct values
        if value is None and num_unique <= np.sqrt(num_data):  # num_data must be > 161 to be true.
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:  # note the alternative is never possible anyway.
                    value = Categorical(name, similarity_based=True)
                    reason = "Small (< sqrt(N)) number of distinct values. "

        # Return non-numeric value and handle NaNs if necessary
        if value is not None:
            return value, reason

        # ========== Numeric value ==========
        # Try parsing if object type
        if col.dtype.kind == 'O':
            numeric_data = pd.to_numeric(col, errors='coerce')
            num_nan = numeric_data.isna().sum()
            if num_nan / num_data < self.config['parsing_nan_fraction_threshold']:
                assert numeric_data.dtype.kind in ('f', 'i')
            else:
                numeric_data = None
        elif col.dtype.kind in ('f', 'i'):
            numeric_data = col

        else:
            numeric_data = None

        # Return numeric value and handle NaNs if necessary
        if numeric_data is not None and numeric_data.dtype.kind in ('f', 'i'):
            value = Float(name)
            reason = f"Converted to numeric dtype ({numeric_data.dtype.kind}) with success " + \
                     f"rate > {1.0 - self.config['parsing_nan_fraction_threshold']}. "

            return value, reason

        # ========== Fallback values ==========

        # Sampling value otherwise
        value = Meta(name)
        reason = "No other criteria met. "

        return value, reason


def _column_does_not_contain_genuine_floats(col: pd.Series) -> bool:
    """Returns TRUE of the input column contains genuine floats, that would exclude integers with type float.

        e.g.:
            _column_does_not_contain_genuine_floats(['A', 'B', 'C']) returns True
            _column_does_not_contain_genuine_floats([1.0, 3.0, 2.0]) returns True
            _column_does_not_contain_genuine_floats([1.0, 3.2, 2.0]) returns False

    :param col: input pd.Series
    :return: bool
    """

    return not col.dropna().apply(_is_not_integer_float).any()


def _is_not_integer_float(x) -> bool:
    """Returns whether 'x' is a float and is not integer.

        e.g.:
            _is_not_integer_float(3.0) = False
            _is_not_integer_float(3.2) = True

    :param x: input
    :return: bool
    """

    if type(x) == float:
        return not x.is_integer()
    else:
        return False


def _is_numeric(col: pd.Series) -> bool:
    """Check whether col contains only numeric values"""
    if col.dtype.kind in ('f', 'i', 'u'):
        return True
    elif col.astype(str).str.isnumeric().all():
        return True
    else:
        return False


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

    def to_json(self):
        raise NotImplementedError

    def from_json(self):
        raise NotImplementedError


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
        if (np.diff(x[self.name]) > 0).all():
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
        super().extract(x)
        if self.date_format is None:
            self.date_format = get_date_format(x[self.name])


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
