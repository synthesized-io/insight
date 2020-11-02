from typing import Type, Union, List, Optional
from collections import defaultdict, OrderedDict

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer


# Can each ValueMeta have multiple transformers?
# Transformers should transform DataFrameMeta too? 
# Then call extract? Quite complex recursive..

class Transformer(TransformerMixin):
    """
    Base class for data transformers.

    Derived classes must implement transform and inverse_
    transform methods.

    Attributes:
        name (str) : the data frame column to transform.
        dtypes (list, optional) : list of valid dtypes for this
          transformation, defaults to None.
    """

    def __init__(self, name: str, dtypes: Optional[List]=None):
        super().__init__()
        self.name = name
        self.dtypes = dtypes
        self._transformers = OrderedDict()

    def register_transformer(self, name: str, transformer: 'Transformer') -> None:
        if not isinstance(transformer, Transformer):
            raise TypeError(f"cannot assign '{type(transformer)}' object to transformer '{name}'",
                    "(synthesized.metadata.transformer.Transformer required)")
        self._transformers[name] = transformer

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Transformer):
            self.register_transformer(name, value)
        object.__setattr__(self, name, value)

    def __call__(self, x: pd.DataFrame, inverse=False) -> pd.DataFrame:
        if not inverse:
            return self.transform(x)
        else:
            return self.inverse_transform(x)

    def fit(self, x: [pd.Series, pd.DataFrame]) -> 'Transformer':
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class QuantileTransformer(Transformer):
    """
    Transform a continuous distribution using the quantile transform.

    Attributes:
        name (str) : the data frame column to transform.
	n_quantiles (int, optional). Number of quantiles to compute, defaults to 1000.
	output_distribution (str). Marginal distribution for the transformed data.
          Either 'uniform' or 'normal', defaults to 'normal'.
    """

    def __init__(self, name: str, n_quantiles: int = 1000, output_distribution: str ='normal', noise: float = 1e-7):
        super().__init__(name=name)
        self._transformer = _QuantileTransformer(n_quantiles, output_distribution)
        self.noise = noise

    def fit(self, x: pd.DataFrame) -> Transformer:
        self._transformer.fit(x[[self.name]])
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.noise:
            x[self.name] += np.random.normal(loc=0, scale=self.noise, size=(len(x)))
        x[self.name] = self._transformer.transform(x[[self.name]])
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = self._transformer.inverse_transform(x[[self.name]])
        return x


class CategoricalTransformer(Transformer):
    """
    Map nominal values into integers.

    Attributes:
        name (str) : the data frame column to transform.
	categories (list, optional). list of unique categories, defaults to None
            If None, categories are extracted from the data.
    """

    def __init__(self, name: str, categories: Optional[List] = None):
        super().__init__(name=name)
        self.categories = categories
        self.idx_to_category = {0: np.nan}
        self.category_to_idx = defaultdict(lambda : 0)

    def fit(self, x: pd.DataFrame) -> 'Transformer':
        if self.categories is None:
            self.categories = x[self.name].unique()

        for idx, cat in enumerate(self.categories):
            self.category_to_idx[cat] = idx+1
            self.idx_to_category[idx+1] = cat

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = x[self.name].apply(lambda x: self.category_to_idx[x])
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = x[self.name].apply(lambda x: self.idx_to_category[x])
        return x

class DateTransformer(Transformer):
    """
    Transform datetime64 field to a continuous representation.
    Values are normalised to a timedelta relative to first datetime in data.
    """

    def __init__(self, name: str, date_format: str = None, unit: str = 'days', start_date: str = None):
        super().__init__(name)
        self.date_format = date_format
        self.unit = unit
        self.start_date = start_date

    def fit(self, x: pd.DataFrame) -> 'Transformer':

        x = x[self.name]
        if x.dtype.kind != 'M':
            x = pd.to_datetime(x, format=self.date_format)

        if self.start_date is None:
            self.start_date = x.min()

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if x[self.name].dtype.kind != 'M':
            x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)
        x[self.name] = (x[self.name] - self.start_date).dt.components[self.unit]
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = (pd.to_timedelta(x[self.name], unit=self.unit) + self.start_date).astype(str)
        return x

class DateCategoricalTransformer(Transformer):
    """
    Creates hour, day-of-week, day and month values from a datetime, and
    transforms using CategoricalTransformer.
    """

    def __init__(self, name: str, date_format: str = None):
        super().__init__(name=name)
        self.date_format = date_format

        self.hour_transform = CategoricalTransformer(f'{self.name}_hour')
        self.dow_transform = CategoricalTransformer(f'{self.name}_dow')
        self.day_transform = CategoricalTransformer(f'{self.name}_day')
        self.month_transform = CategoricalTransformer(f'{self.name}_month')

    def fit(self, x: pd.DataFrame) -> 'Transformer':

        x = x[self.name]
        x = self.split_datetime(x)

        for transformer in self._transformers.values():
            transformer.fit(x)

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        if x[self.name].dtype.kind != 'M':
            x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)

        x = pd.concat((x, self.split_datetime(x[self.name])), axis=1)

        for transformer in self._transformers.values():
            x = transformer.transform(x)
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x.drop(columns=[f'{self.name}_hour', f'{self.name}_dow',
                f'{self.name}_day', f'{self.name}_month'], inplace=True)
        return x

    def split_datetime(self, col: pd.Series) -> pd.DataFrame:
        """Split datetime column into separate hour, dow, day and month fields."""

        if col.dtype.kind != 'M':
            col = pd.to_datetime(col, format=self.date_format)

        return pd.DataFrame({
            f'{col.name}_hour': col.dt.hour,
            f'{col.name}_dow': col.dt.weekday,
            f'{col.name}_day': col.dt.day,
            f'{col.name}_month': col.dt.month
        })




#class NanTransformer(Transformer):
#    """Creates a Boolean field to indicate the presence of a NaN value."""
#
#    def __init__


