import functools
from typing import Any, Optional, Union
from dataclasses import asdict

import numpy as np
import pandas as pd

from .meta import Meta, ValueMeta, Categorical, Constant, Bool, Date, TimeDelta, Integer, Float, Ordinal, Nominal
from .meta import DataFrameMeta
from .meta import get_date_format
from .exceptions import UnknownDateFormatError, UnsupportedDtypeError
from ..config import MetaFactoryConfig


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
    def wrapper(cls, sr: pd.Series) -> Union['Constant', 'Categorical', 'Meta']:
        n_unique = sr.nunique()
        if n_unique == 1:
            return Constant(str(sr.name))
        elif n_unique <= max(cls.min_num_unique, cls.categorical_threshold_log_multiplier * np.log(len(sr))) \
                and (not _MetaBuilder._contains_genuine_floats(sr)):
            return Categorical(str(sr.name), similarity_based=True if n_unique > 2 else False)
        else:
            return func(cls, sr, **cls.kwargs)
    return wrapper


class _MetaBuilder():
    """
    A functor class used internally by MetaFactory.

    Implements methods that return a derived Meta instance for a given pd.Series.
    The underyling numpy dtype (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
    determines the method that is called, and therefore the Meta that is returned.
    """
    def __init__(self, min_num_unique: int, parsing_nan_fraction_threshold: float, categorical_threshold_log_multiplier: float, **kwargs):
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
        self.parsing_nan_fraction_threshold = parsing_nan_fraction_threshold
        self.categorical_threshold_log_multiplier = categorical_threshold_log_multiplier

        self.kwargs = kwargs

    def __call__(self, sr: pd.Series) -> ValueMeta:
        return self._dtype_builders[sr.dtype.kind](sr, **self.kwargs)

    def _DateBuilder(self, sr: pd.Series, **kwargs) -> Date:
        return Date(str(sr.name), **kwargs)

    def _TimeDeltaBuilder(self, sr: pd.Series, **kwargs) -> TimeDelta:
        return TimeDelta(str(sr.name), **kwargs)

    def _BoolBuilder(self, sr: pd.Series, **kwargs) -> Bool:
        return Bool(str(sr.name), **kwargs)

    @_default_categorical
    def _IntBuilder(self, sr: pd.Series, **kwargs) -> Integer:
        return Integer(str(sr.name), **kwargs)

    @_default_categorical
    def _FloatBuilder(self, sr: pd.Series, **kwargs) -> Union[Float, Integer]:

        # check if is integer (in case NaNs which cast to float64)
        # delegate to __IntegerBuilder
        if self._contains_genuine_floats(sr):
            return Float(str(sr.name), **kwargs)
        else:
            return self._IntBuilder(sr, **kwargs)

    def _CategoricalBuilder(self, sr: pd.Series, **kwargs) -> Union[Ordinal, Categorical]:
        if isinstance(sr.dtype, pd.CategoricalDtype):
            categories = sr.cat.categories.tolist()
            if sr.cat.ordered:
                return Ordinal(str(sr.name), categories=categories, **kwargs)
            else:
                return Categorical(str(sr.name), categories=categories, **kwargs)

        else:
            return Categorical(str(sr.name), **kwargs)

    def _ObjectBuilder(self, sr: pd.Series, **kwargs) -> Union[Nominal, Date, Categorical, Float, Integer]:
        try:
            get_date_format(sr)
            return self._DateBuilder(sr, **kwargs)
        except (UnknownDateFormatError, ValueError, TypeError, OverflowError):

            n_unique = sr.nunique()
            n_rows = len(sr)

            x_numeric = pd.to_numeric(sr, errors='coerce')
            num_nan = x_numeric.isna().sum()

            if isinstance(sr.dtype, pd.CategoricalDtype):
                return self._CategoricalBuilder(sr, similarity_based=True if n_unique > 2 else False)

            elif (n_unique <= np.sqrt(n_rows)
                  or n_unique <= max(self.min_num_unique, self.categorical_threshold_log_multiplier * np.log(len(sr)))) \
                    and (not self._contains_genuine_floats(x_numeric)):
                return self._CategoricalBuilder(sr, similarity_based=True if n_unique > 2 else False)

            elif num_nan / n_rows < self.parsing_nan_fraction_threshold:
                if self._contains_genuine_floats(x_numeric):
                    return self._FloatBuilder(x_numeric)
                else:
                    return self._IntBuilder(x_numeric)

            else:
                return Nominal(str(sr.name))

    @staticmethod
    def _contains_genuine_floats(sr: pd.Series) -> bool:
        return (~sr.dropna().apply(_MetaBuilder._is_integer_float)).any()

    @staticmethod
    def _is_integer_float(x: Any) -> bool:
        """Returns True if x can be represented as an integer."""
        try:
            return float(x).is_integer()
        except (ValueError, TypeError):
            return False


class MetaFactory():
    """Factory class to create Meta instances from pd.Series and pd.DataFrame objects."""
    def __init__(self, config: Optional[MetaFactoryConfig] = None):

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

    def _from_series(self, sr: pd.Series) -> ValueMeta:
        if sr.dtype.kind not in self._builder._dtype_builders:
            raise UnsupportedDtypeError(f"'{sr.dtype}' is unsupported")
        return self._builder(sr)

    def _from_df(self, df: pd.DataFrame, name: Optional[str] = 'df') -> DataFrameMeta:
        if name is None:
            raise ValueError("name must not be a string, not None")
        meta = DataFrameMeta(name)
        for col in df.columns:
            try:
                child = self._from_series(df[col])
                meta[child.name] = child
            except TypeError as e:
                print(f"Warning. Encountered error when interpreting ValueMeta for '{col}'", e)
        return meta

    @staticmethod
    def default_config() -> MetaFactoryConfig:
        return MetaFactoryConfig()


class MetaExtractor(MetaFactory):
    """Extract the DataFrameMeta for a data frame"""
    def __init__(self, config: Optional[MetaFactoryConfig] = None):
        super().__init__(config)

    @staticmethod
    def extract(df: pd.DataFrame, config: Optional[MetaFactoryConfig] = None) -> DataFrameMeta:
        """
        Instantiate and extract the DataFrameMeta that describes a data frame.

        Args:
            df: the data frame to instantiate and extract DataFrameMeta.
            config: Optional; The configuration parameters to MetaFactory.

        Returns:
            A DataFrameMeta instance for which all child meta have been extracted.

        Raises:
            UnsupportedDtypeError: The data type of a column in the data frame pandas is not supported.
            TypeError: An error occured during instantiation of a ValueMeta.
        """

        factory = MetaExtractor(config)
        df_meta = factory._from_df(df).extract(df)
        return df_meta
