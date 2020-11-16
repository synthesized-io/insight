import functools
from typing import Any, Optional, Union
from dataclasses import asdict

import numpy as np
import pandas as pd

from .meta import Meta, ValueMeta, Categorical, Constant, Bool, Date, TimeDelta, Integer, Float, Ordinal, Nominal
from .meta import DataFrameMeta
from .meta import get_date_format
from .exceptions import UnknownDateFormatError, UnsupportedDtypeError
from ..config import MetaExtractorConfig


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
            return Constant(str(x.name))
        elif n_unique <= max(cls.min_num_unique, cls.categorical_threshold_log_multiplier * np.log(len(x))) \
                and (not _MetaBuilder._contains_genuine_floats(x)):
            return Categorical(str(x.name), similarity_based=True if n_unique > 2 else False)
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
        return Date(str(x.name), **kwargs)

    def _TimeDeltaBuilder(self, x: pd.Series, **kwargs) -> TimeDelta:
        return TimeDelta(str(x.name), **kwargs)

    def _BoolBuilder(self, x: pd.Series, **kwargs) -> Bool:
        return Bool(str(x.name), **kwargs)

    @_default_categorical
    def _IntBuilder(self, x: pd.Series, **kwargs) -> Integer:
        return Integer(str(x.name), **kwargs)

    @_default_categorical
    def _FloatBuilder(self, x: pd.Series, **kwargs) -> Union[Float, Integer]:

        # check if is integer (in case NaNs which cast to float64)
        # delegate to __IntegerBuilder
        if self._contains_genuine_floats(x):
            return Float(str(x.name), **kwargs)
        else:
            return self._IntBuilder(x, **kwargs)

    def _CategoricalBuilder(self, x: pd.Series, **kwargs) -> Union[Ordinal, Categorical]:
        if isinstance(x.dtype, pd.CategoricalDtype):
            categories = x.cat.categories.tolist()
            if x.cat.ordered:
                return Ordinal(str(x.name), categories=categories, **kwargs)
            else:
                return Categorical(str(x.name), categories=categories, **kwargs)

        else:
            return Categorical(str(x.name), **kwargs)

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

            elif (n_unique <= np.sqrt(n_rows) or
                  n_unique <= max(self.min_num_unique, self.categorical_threshold_log_multiplier * np.log(len(x)))) \
                    and (not self._contains_genuine_floats(x_numeric)):
                return self._CategoricalBuilder(x, similarity_based=True if n_unique > 2 else False)

            elif num_nan / n_rows < self.acceptable_nan_frac:
                if self._contains_genuine_floats(x_numeric):
                    return self._FloatBuilder(x_numeric)
                else:
                    return self._IntBuilder(x_numeric)

            else:
                return Nominal(str(x.name))

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
            self.config = MetaFactory.default_config()
        else:
            self.config = config

        self._builder = _MetaBuilder(**asdict(self.config))

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
        return MetaExtractorConfig()


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
        df_meta = factory._from_df(x)
        df_meta.extract(x)
        return df_meta
