from typing import List, Optional
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

from .meta import Meta, DataFrameMeta, Affine, ValueMeta


class TransformerNotFitError(Exception):
    pass


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

    def __init__(self, name: str, dtypes: Optional[List] = None):
        super().__init__()
        self.name = name
        self.dtypes = dtypes
        self._transformers = None
        self._fitted = False

    def __repr__(self):

        nl = '\n    '
        if self.transformers:
            return f'{self.__class__.__name__}({nl}{nl.join([t.__repr__() for t in self.transformers])})'
        else:
            return f'{self.__class__.__name__}({self.name})'

    @property
    def transformers(self) -> List['Transformer']:
        return self._transformers

    def register_transformer(self, name: str, transformer: 'Transformer') -> None:
        if not isinstance(transformer, Transformer):
            raise TypeError(f"cannot assign '{type(transformer)}' object to transformer '{name}'",
                            "(synthesized.metadata.transformer.Transformer required)")
        if self._transformers is None:
            self._transformers = []
        self._transformers.append(transformer)

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
        if not self._fitted:
            self._fitted = True
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        for transformer in self.transformers:
            transformer = transformer.fit(x)
            x = transformer.transform(x)
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        for name, transformer in reversed(self.transformers):
            x = transformer.inverse_transform(x)
        return x

    @classmethod
    def from_meta(cls, meta: ValueMeta, **kwargs):
        return cls(meta.name, **kwargs)


class QuantileTransformer(Transformer):
    """
    Transform a continuous distribution using the quantile transform.

    Attributes:
        name (str) : the data frame column to transform.
        n_quantiles (int, optional). Number of quantiles to compute, defaults to 1000.
        output_distribution (str). Marginal distribution for the transformed data.
            Either 'uniform' or 'normal', defaults to 'normal'.
    """

    def __init__(self, name: str, n_quantiles: int = 1000, output_distribution: str = 'normal', noise: float = 1e-7):
        super().__init__(name=name)
        self._transformer = _QuantileTransformer(n_quantiles, output_distribution)
        self.noise = noise

    def fit(self, x: pd.DataFrame) -> Transformer:
        if len(x) < self._transformer.n_quantiles:
            self._transformer = self._transformer.set_params(n_quantiles=len(x))

        if self.noise:
            x[self.name] += np.random.normal(loc=0, scale=self.noise, size=(len(x)))

        self._transformer.fit(x[[self.name]])
        return super().fit(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.noise:
            x[self.name] += np.random.normal(loc=0, scale=self.noise, size=(len(x)))

        positive = (x[self.name] > 0.0).all()
        nonnegative = (x[self.name] >= 0.0).all()

        if nonnegative and not positive:
            x[self.name] = np.maximum(x[self.name], 0.001)

        x[self.name] = self._transformer.transform(x[[self.name]])

        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = self._transformer.inverse_transform(x[[self.name]])
        return x

    @classmethod
    def from_meta(cls, meta, **kwargs):
        return cls(meta.name, **kwargs)


class CategoricalTransformer(Transformer):
    """
    Map nominal values onto integers.

    Attributes:
        name (str) : the data frame column to transform.
        categories (list, optional). list of unique categories, defaults to None
            If None, categories are extracted from the data.
    """

    def __init__(self, name: str, categories: Optional[List] = None):
        super().__init__(name=name)
        self.categories = categories
        self.idx_to_category = {0: np.nan}
        self.category_to_idx = defaultdict(lambda: 0)

    def fit(self, x: pd.DataFrame) -> 'Transformer':

        if self.categories is None:
            categories = x[self.name].unique()
        else:
            categories = np.array(self.categories)

        try:
            categories.sort()
        except TypeError:
            pass

        # check for NaN and delete to put at front of category array
        try:
            categories = np.delete(categories, np.isnan(categories))
        except TypeError:
            pass
        categories = np.array([np.nan, *categories])

        for idx, cat in enumerate(categories[1:]):
            self.category_to_idx[cat] = idx + 1
            self.idx_to_category[idx + 1] = cat

        return super().fit(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        # convert NaN to str. Otherwise np.nan are used as dict keys, which can be dodgy
        x.loc[x[self.name].isna(), self.name] = x[self.name].astype(str)
        x[self.name] = x[self.name].apply(lambda x: self.category_to_idx[x])
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = x[self.name].apply(lambda x: self.idx_to_category[x])
        return x

    @classmethod
    def from_meta(cls, meta):
        return cls(meta.name, meta.categories)


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

        if self.date_format is None:
            self.date_format = get_date_format(x)
        if x.dtype.kind != 'M':
            x = pd.to_datetime(x, format=self.date_format)
        if self.start_date is None:
            self.start_date = x.min()

        return super().fit(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if x[self.name].dtype.kind != 'M':
            x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)
        x[self.name] = (x[self.name] - self.start_date).dt.components[self.unit]
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = (pd.to_timedelta(x[self.name], unit=self.unit) + self.start_date).apply(lambda x: x.strftime(self.date_format))
        return x

    @classmethod
    def from_meta(cls, meta, **kwargs):
        return cls(meta.name, meta.date_format, start_date=meta.min, **kwargs)


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
        if self.date_format is None:
            self.date_format = get_date_format(x)
        x = self.split_datetime(x)

        for transformer in self.transformers:
            transformer.fit(x)

        return super().fit(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        if x[self.name].dtype.kind != 'M':
            x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)

        x = pd.concat((x, self.split_datetime(x[self.name])), axis=1)

        for transformer in self.transformers:
            x = transformer.transform(x)
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x.drop(columns=[f'{self.name}_hour', f'{self.name}_dow',
                        f'{self.name}_day', f'{self.name}_month'], inplace=True)
        return x

    @classmethod
    def from_meta(cls, meta, **kwargs):
        return cls(meta.name, meta.date_format, **kwargs)

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


class NanTransformer(Transformer):
    """Creates a Boolean field to indicate the presence of a NaN value in an affine field."""

    def __init__(self, name: str):
        super().__init__(name)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[f'{self.name}_nan'] = x[self.name].isna().astype(int)
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x.drop(columns=[f'{self.name}_nan'], inplace=True)
        return x

    @classmethod
    def from_meta(cls, meta: ValueMeta) -> 'Transformer':
        return cls(meta.name)


class TransformerFactory():
    def __init__(self, transformer_config=None):

        if transformer_config is None:
            self.config = meta_transformer_config
        else:
            self.config = transformer_config

    def create_transformers(self, meta: Meta) -> Transformer:

        if isinstance(meta, DataFrameMeta):
            transformer = DataFrameTransformer(meta, meta.name)
            for name, meta in meta.column_meta.items():
                children = self.create_transformers(meta)
                for child in children:
                    setattr(transformer, child.name, child)
        else:
            transformer = self._from_meta(meta)
        return transformer

    def _from_meta(self, meta: Meta) -> Transformer:
        transformers = []
        meta_transformers = self.config['meta'][meta.__class__.__name__]
        for idx, transformer in enumerate(meta_transformers):
            try:
                kwargs = self.config[f'{meta.name}.{transformer}']
            except KeyError:
                kwargs = {}
            transformers.append(transformer.from_meta(meta, **kwargs))
            if isinstance(meta, Affine) and meta.nan_freq > 0:
                transformers.append(NanTransformer.from_meta(meta))
        return transformers


class DataFrameTransformer(Transformer):

    def __init__(self, meta: DataFrameMeta, name='df'):
        super().__init__(name)
        self.meta = meta


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


meta_transformer_config = {
    'meta': {
        'Float': [QuantileTransformer],
        'Integer': [QuantileTransformer],
        'Bool': [CategoricalTransformer],
        'Categorical': [CategoricalTransformer],
        'Date': [DateCategoricalTransformer, DateTransformer, QuantileTransformer]
    },
    'Float.QuantileTransformer': {
        'n_quantiles': 1000,
        'distribution': 'normal'
    },
    'Integer.QuantileTransformer': {
        'n_quantiles': 1000,
        'distribution': 'normal'
    },
    'Date.DateTransformer': {
        'unit': 'days'
    },
    'Date.QuantileTransformer': {
        'n_quantiles': 1000,
        'distribution': 'normal'
    },
}
