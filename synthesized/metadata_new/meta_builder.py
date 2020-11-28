from typing import Any, Optional, Union, Callable, Dict
from dataclasses import asdict

import numpy as np
import pandas as pd

from .base import ValueMeta
from .data_frame_meta import DataFrameMeta
from .datetime import Date, TimeDelta, get_date_format
from .bool import Bool
from .categorical import String
from .continuous import Integer, Float
from .ordinal import OrderedString

from .exceptions import UnknownDateFormatError, UnsupportedDtypeError
from ..config import MetaFactoryConfig


class _MetaBuilder:
    """
    A functor class used internally by MetaFactory.

    Implements methods that return a derived Meta instance for a given pd.Series.
    The underyling numpy dtype (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
    determines the method that is called, and therefore the Meta that is returned.
    """
    def __init__(
            self, min_num_unique: int, parsing_nan_fraction_threshold: float,
            categorical_threshold_log_multiplier: float
    ):
        self._dtype_builders: Dict[str, Callable[[pd.Series], ValueMeta[Any]]] = {
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

    def __call__(self, sr: pd.Series) -> ValueMeta[Any]:
        return self._dtype_builders[sr.dtype.kind](sr)

    def _DateBuilder(self, sr: pd.Series) -> Date:
        return Date(str(sr.name))

    def _TimeDeltaBuilder(self, sr: pd.Series) -> TimeDelta:
        return TimeDelta(str(sr.name))

    def _BoolBuilder(self, sr: pd.Series,) -> Bool:
        return Bool(str(sr.name))

    def _IntBuilder(self, sr: pd.Series) -> Integer:
        return Integer(str(sr.name))

    def _FloatBuilder(self, sr: pd.Series) -> Union[Float, Integer]:

        # check if is integer (in case NaNs which cast to float64)
        # delegate to __IntegerBuilder
        if self._contains_genuine_floats(sr):
            return Float(str(sr.name))
        else:
            return self._IntBuilder(sr)

    def _CategoricalBuilder(self, sr: pd.Series) -> Union[OrderedString, String]:
        if isinstance(sr.dtype, pd.CategoricalDtype):
            if sr.cat.ordered:
                return OrderedString(str(sr.name))
            else:
                return String(str(sr.name))

        else:
            return String(str(sr.name))

    def _ObjectBuilder(self, sr: pd.Series) -> Union[Date, String, OrderedString, Float, Integer]:
        try:
            get_date_format(sr)
            return self._DateBuilder(sr)
        except (UnknownDateFormatError, ValueError, TypeError, OverflowError):

            n_unique = sr.nunique()
            n_rows = len(sr)

            x_numeric = pd.to_numeric(sr, errors='coerce')
            num_nan = x_numeric.isna().sum()

            if isinstance(sr.dtype, pd.CategoricalDtype):
                return self._CategoricalBuilder(sr)

            elif (n_unique <= np.sqrt(n_rows)
                  or n_unique <= max(self.min_num_unique, self.categorical_threshold_log_multiplier * np.log(len(sr))))\
                    and (not self._contains_genuine_floats(x_numeric)):
                return self._CategoricalBuilder(sr)

            elif num_nan / n_rows < self.parsing_nan_fraction_threshold:
                if self._contains_genuine_floats(x_numeric):
                    return self._FloatBuilder(x_numeric)
                else:
                    return self._IntBuilder(x_numeric)

            else:
                return String(str(sr.name))

    @staticmethod
    def _contains_genuine_floats(sr: pd.Series) -> bool:
        b: bool = (~sr.dropna().apply(_MetaBuilder._is_integer_float)).any()
        return b

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

    def __call__(self, x: Union[pd.Series, pd.DataFrame]) -> Union[ValueMeta[Any], DataFrameMeta]:
        return self.create_meta(x)

    def create_meta(
            self, x: Union[pd.Series, pd.DataFrame], name: Optional[str] = 'df'
    ) -> Union[ValueMeta[Any], DataFrameMeta]:
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

    def _from_series(self, sr: pd.Series) -> ValueMeta[Any]:
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
