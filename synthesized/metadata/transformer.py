from typing import List, Optional, Union, Dict, TypeVar, Type
from collections import defaultdict
from datetime import datetime
import functools
import logging
import copy

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

from .meta import Meta, DataFrameMeta, Nominal, Affine, ValueMeta, Date, Float, Integer, get_date_format
from .exceptions import *

logger = logging.getLogger(__name__)

TransformerType = TypeVar('TransformerType', bound='Transformer')
class Transformer(TransformerMixin):
    """
    Base class for data frame transformers.

    Derived classes must implement transform. The
    fit method is optional, and should be used to
    extract required transform parameters from the data.

    Attributes:
        name: the data frame column to transform.

        dtypes: list of valid dtypes for this
          transformation, defaults to None.
    """

    def __init__(self, name: str, dtypes: Optional[List] = None):
        super().__init__()
        self.name = name
        self.dtypes = dtypes
        self._transformers: List['Transformer'] = []
        self._fitted = False

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    def __add__(self, other: 'Transformer') -> 'SequentialTransformer':
        return SequentialTransformer(name=self.name, transformers=[self, other])

    def __call__(self, x: pd.DataFrame, inverse=False) -> pd.DataFrame:
        if not inverse:
            return self.transform(x)
        else:
            return self.inverse_transform(x)

    def fit(self: TransformerType, x: Union[pd.Series, pd.DataFrame]) -> TransformerType:
        if not self._fitted:
            self._fitted = True
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
         raise NotImplementedError

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NonInvertibleTransformError

    @classmethod
    def from_meta(cls: Type[TransformerType], meta, **kwargs) -> TransformerType:
        raise NotImplementedError


class SequentialTransformer(Transformer):
    """
    Transform data using a sequence of pre-defined Transformers.

    Each transformer can act on different columns of a data frame,
    or the same column. In the latter case, each transformer in
    the sequence is fit to the transformed data from the previous.

    Attributes:
        name: the data frame column to transform.

        transformers: list of Transformers

        dtypes: Optional; list of valid dtypes for this
          transformation, defaults to None.

    Examples:

        Create the data to transform:

        >>> df = pd.DataFrame({'x': ['A', 'B', 'C'], 'y': [0, 10, np.nan]})
            x     y
        0   A   0.0
        1   B   10.0
        2   C   NaN

        Define a SequentialTransformer:

        >>> t = SequentialTransformer(
                    name='t',
                    transformers=[
                        CategoricalTransformer('x'),
                        NanTransformer('y'),
                        QuantileTransformer('y')
                    ]
                )

        Transform the data frame:

        >>> df_transformed = t.transform(df)
            x    y     y_nan
        0   1   -5.19    0
        1   2    5.19    0
        2   3    NaN     1

        Alternatively, transformers can be bound as attributes:

        >>> t = SequentialTransformer(name='t')
        >>> t.x_transform = CategoricalTransformer('x')
        >>> t.y_transform = SequentialTransformer(
                                name='y_transform',
                                transformers=[
                                    NanTransformer('y'),
                                    QuantileTransformer('y')
                                ]
                            )

        Transform the data frame:

        >>> df_transformed = t.transform(df)
            x    y     y_nan
        0   1   -5.19    0
        1   2    5.19    0
        2   3    NaN     1
    """
    def __init__(self, name: str, transformers: Optional[List[Transformer]] = None, dtypes: Optional[List] = None):

        super().__init__(name, dtypes)

        if not transformers:
            transformers = []

        self.transformers = transformers

    def __iter__(self):
        yield from self.transformers

    def __getitem__(self, idx):
        return self.transformers[idx]

    def __repr__(self):
        nl = '\n    '
        return f'{self.__class__.__name__}({nl}{nl.join([repr(t) for t in self.transformers])})'

    def __add__(self, other: 'Transformer') -> 'SequentialTransformer':
        if isinstance(other, SequentialTransformer):
            return SequentialTransformer(name=self.name, transformers=self.transformers + other.transformers)
        else:
            return SequentialTransformer(name=self.name, transformers=self.transformers + [other])

    def _group_transformers_by_name(self) -> Dict[str, List[Transformer]]:
        d = defaultdict(list)
        for transfomer in self:
            d[transfomer.name].append(transfomer)
        return d

    @property
    def transformers(self) -> List[Transformer]:
        """Retrieve the sequence of Transformers."""
        return self._transformers

    @transformers.setter
    def transformers(self, value: List[Transformer]) -> None:
        """Set the sequence of Transformers."""
        for transformer in value:
            self.add_transformer(transformer)

    def add_transformer(self, transformer: Transformer) -> None:
        if not isinstance(transformer, Transformer):
            raise TypeError(f"cannot add '{type(transformer)}' as a transformer.",
                            "(synthesized.metadata.transformer.Transformer required)")
        self._transformers.append(transformer)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Perform the sequence of transformations."""
        for group, transformers in self._group_transformers_by_name().items():
            for t in transformers:
                x = t.fit_transform(x)
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Invert the sequence of transformations, if possible."""
        for group, transformers in self._group_transformers_by_name().items():
            for t in reversed(transformers):
                try:
                    x = t.inverse_transform(x)
                except NonInvertibleTransformError:
                    logger.warning("Encountered transform with no inverse. ")
                finally:
                    x = x
        return x

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Transformer):
            self.add_transformer(value)
        object.__setattr__(self, name, value)


class HistogramTransformer(Transformer):
    """
    Bin continous values into discrete bins.

    Attributes:
        name: the data frame column to transform.

        bins: the number of equally spaced bins or a predefined list of bin edges.

        **kwargs: keyword arguments to pd.cut

    See also:
        pd.cut
    """
    def __init__(self, name: str, bins: Union[List, int], inplace=True, **kwargs):
        super().__init__(name)
        self.bins = bins
        self.kwargs = kwargs
        self.inplace = inplace

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = pd.cut(x[self.name], self.bins, **self.kwargs)
        return x


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
    def from_meta(cls, meta: Union[Float, Integer], **kwargs) -> 'QuantileTransformer':
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
        self.category_to_idx: Dict[str, int] = defaultdict(lambda: 0)

    def fit(self, x: pd.DataFrame) -> Transformer:

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
    def from_meta(cls, meta: Nominal, **kwargs) -> 'CategoricalTransformer':
        return cls(meta.name, meta.categories)


class DateTransformer(Transformer):
    """
    Transform datetime64 field to a continuous representation.
    Values are normalised to a timedelta relative to first datetime in data.

    Attributes:
        name: the data frame column to transform.

        date_format: Optional; string representation of date format, eg. '%d/%m/%Y'.

        unit: Optional; unit of timedelta.

        start_date: Optional; when normalising dates, compare relative to this.
    """

    def __init__(self, name: str, date_format: str = None, unit: str = 'days', start_date: Optional[pd.Timestamp] = None):
        super().__init__(name)
        self.date_format = date_format
        self.unit = unit
        self.start_date = start_date

    def fit(self, x: pd.DataFrame) ->  Transformer:

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
    def from_meta(cls, meta: Date, **kwargs) -> 'DateTransformer':
        return cls(meta.name, meta.date_format, start_date=meta.min)


class DateCategoricalTransformer(SequentialTransformer):
    """
    Creates hour, day-of-week, day and month values from a datetime, and
    transforms using CategoricalTransformer.

    Attributes:
        name: the data frame column to transform.

        date_format: Optional; string representation of date format, eg. '%d/%m/%Y'.
    """

    def __init__(self, name: str, date_format: str = None):
        super().__init__(name=name)
        self.date_format = date_format

        self.hour_transform = CategoricalTransformer(f'{self.name}_hour')
        self.dow_transform = CategoricalTransformer(f'{self.name}_dow')
        self.day_transform = CategoricalTransformer(f'{self.name}_day')
        self.month_transform = CategoricalTransformer(f'{self.name}_month')

    def fit(self, x: pd.DataFrame) -> Transformer:

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

        categorical_dates = self.split_datetime(x[self.name])
        x[categorical_dates.columns] = categorical_dates

        return super().transform(x)

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x.drop(columns=[f'{self.name}_hour', f'{self.name}_dow',
                        f'{self.name}_day', f'{self.name}_month'], inplace=True)
        return x

    @classmethod
    def from_meta(cls, meta: Date, **kwargs) -> 'DateCategoricalTransformer':
        return cls(meta.name, meta.date_format)

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
    """
    Creates a Boolean field to indicate the presence of a NaN value in an affine field.

    Attributes:
        name: the data frame column to transform.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[f'{self.name}_nan'] = x[self.name].isna().astype(int)
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x.drop(columns=[f'{self.name}_nan'], inplace=True)
        return x

    @classmethod
    def from_meta(cls, meta: Affine, **kwargs) -> 'NanTransformer':
        return cls(meta.name)


class DataFrameTransformer(SequentialTransformer):
    """
    Transform a data frame.

    This is a SequentialTransform built from a DataFrameMeta instance.

    Attributes:
        meta: DataFrameMeta instance returned from MetaExtractor.extract
    """
    def __init__(self, meta: DataFrameMeta, name: Optional[str] = 'df'):
        if name is None:
            raise ValueError("name must not be a string, not None")
        super().__init__(name)
        self.meta = meta

    @classmethod
    def from_meta(cls, meta: DataFrameMeta, **kwargs) -> 'DataFrameTransformer':
        obj = TransformerFactory(**kwargs).create_transformers(meta)
        return obj


class TransformerFactory():
    def __init__(self, transformer_config: Optional[dict] = None):

        if transformer_config is None:
            self.config = meta_transformer_config
        else:
            self.config = transformer_config

    def create_transformers(self, meta: Meta) -> Transformer:

        if isinstance(meta, DataFrameMeta):
            transformer = DataFrameTransformer(meta, meta.name)
            for m in meta.children:
                transformer.add_transformer(self._from_meta(m))
        else:
            return self._from_meta(meta)
        return transformer

    def _from_meta(self, meta: Meta) -> Transformer:
        try:
            meta_transformer = self.config['meta'][meta.__class__.__name__]
        except KeyError:
            raise UnsupportedMetaError(f"{meta.__class__.__name__} has no associated Transformer")

        if isinstance(meta_transformer, list):
            transformer = SequentialTransformer(f'Sequential({meta.name})')
            for t in meta_transformer:
                kwargs = self._get_transformer_kwargs(meta, t)
                transformer.add_transformer(t.from_meta(meta, **kwargs))
        else:
            transformer = meta_transformer.from_meta(meta, **self._get_transformer_kwargs(meta, meta_transformer))

        if isinstance(meta, Affine) and meta.nan_freq > 0:
            transformer += NanTransformer.from_meta(meta)

        return transformer

    def _get_transformer_kwargs(self, meta: Meta, transformer: str) -> dict:
        try:
            return self.config[f'{meta.name}.{transformer}']
        except KeyError:
            return {}

meta_transformer_config: Dict[str, Dict] = {
    'meta': {
        'Float': QuantileTransformer,
        'Integer': QuantileTransformer,
        'Bool': CategoricalTransformer,
        'Categorical': CategoricalTransformer,
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
