from typing import Type, Union, List, Optional
from collections import defaultdict

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
    Transform datetime fields.

    Extracts hour, day-of-week, day and month from a datetime
    """

    def __init__(self, name: str, date_format: str = None,
            continuous_transformer: Type[Transformer] = QuantileTransformer,
            continuous_transformer_kwargs: dict = {},
            categorical_transformer: Type[Transformer] = CategoricalTransformer,
            categorical_transformer_kwargs: dict = {}):
        super().__init__(name=name)
        self.date_format = date_format
        self.transformers = {
                f'{self.name}': continuous_transformer(f'{self.name}', **continuous_transformer_kwargs),
                f'{self.name}_hour': categorical_transformer(f'{self.name}_hour', **categorical_transformer_kwargs),
                f'{self.name}_dow': categorical_transformer(f'{self.name}_dow', **categorical_transformer_kwargs),
                f'{self.name}_day': categorical_transformer(f'{self.name}_day', **categorical_transformer_kwargs),
                f'{self.name}_month': categorical_transformer(f'{self.name}_month', **categorical_transformer_kwargs),
        }

    def fit(self, x: pd.DataFrame) -> 'Transformer':
        x = x[self.name]
        if x.dtype.kind != 'M':
            x = pd.to_datetime(x, self.date_format)
        date_df = self.split_datetime(x)
        date_df[self.name] = self.normalise_datetime(x)
        for transformer in self.transformers.values():
            transformer.fit(date_df)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)
        x = pd.concat((x, self.split_datetime(x[self.name])), axis=1)
        return x
        x[self.name] = self.normalise_datetime(x[self.name])
        return x
        for transformer in self.transformers.values():
            x = transformer.transform(x)
        return x

    @staticmethod
    def normalise_datetime(col: pd.Series) -> pd.Series:
        """Normalise datetime series to number of days relative to the first date"""
        return (col - col.min()).dt.days

    @staticmethod
    def split_datetime(col: pd.Series) -> pd.DataFrame:
        """Split datetime column into separate hour, dow, day and month fields."""

        if col.dtype.kind != 'M':
            col = pd.to_datetime(col)

        return pd.DataFrame({
            f'{col.name}': col,
            f'{col.name}_hour': col.dt.hour,
            f'{col.name}_dow': col.dt.weekday,
            f'{col.name}_day': col.dt.day,
            f'{col.name}_month': col.dt.month
        })




#class NanTransformer(Transformer):
#    """Creates a Boolean field to indicate the presence of a NaN value."""
#
#    def __init__


