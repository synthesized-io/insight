import logging
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import (Any, Callable, Collection, Dict, Iterator, List, MutableSequence, Optional, Tuple, Type, TypeVar,
                    Union)

import numpy as np
import pandas as pd

from .exceptions import NonInvertibleTransformError, TransformerNotFitError
from ..util import get_all_subclasses

logger = logging.getLogger(__name__)

TransformerType = TypeVar('TransformerType', bound='Transformer')


class Transformer:
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
        self._fitted = False

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, dtypes={self.dtypes})'

    def __eq__(self, other):
        def attrs(x):
            return dict(filter(lambda x: not x[0].startswith('_'), x.__dict__.items()))
        try:
            np.testing.assert_equal(attrs(self), attrs(other))
            return True
        except AssertionError:
            return False

    def __call__(self, x: pd.DataFrame, inverse=False, **kwargs) -> pd.DataFrame:
        self._assert_fitted()
        if not inverse:
            return self.transform(x, **kwargs)
        else:
            return self.inverse_transform(x, **kwargs)

    def fit(self: TransformerType, x: Union[pd.Series, pd.DataFrame]) -> TransformerType:
        if not self._fitted:
            self._fitted = True
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NonInvertibleTransformError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fit(df).transform(df)
        return df

    def _assert_fitted(self):
        if not self._fitted:
            raise TransformerNotFitError("Transformer not fitted yet, please call 'fit()' before calling transform.")

    @classmethod
    def from_meta(cls: Type[TransformerType], meta: Any) -> TransformerType:
        """
        Construct a Transformer from a transformer class name and a meta. This is an abstract class, must be
        implemented in each transformer subclass.
        """
        raise NotImplementedError

    @classmethod
    def from_name_and_meta(cls, class_name: str, meta: Any) -> 'Transformer':
        """
        Construct a Transformer from a transformer class name and a meta.

        See also:
            Transformer.from_meta: construct a Transformer from a meta
        """

        registy = cls.get_registry()
        if class_name not in registy.keys():
            raise ValueError(f"Given transformer {class_name} not found in Transformer subclasses.")

        return registy[class_name].from_meta(meta)

    @classmethod
    def get_registry(cls: Type[TransformerType]) -> Dict[str, Type[TransformerType]]:
        return {sc.__name__: sc for sc in get_all_subclasses(cls)}

    @property
    def in_columns(self) -> List[str]:
        return [self.name]

    @property
    def out_columns(self) -> List[str]:
        return [self.name]


class BagOfTransformers(Transformer, Collection[Transformer]):
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

        Define a BagOfTransformers:
        >>> t = BagOfTransformers(
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

        Alternatively, transformers can be appended (or added)
        >>> t = BagOfTransformers(name='t')
        >>> t.append(CategoricalTransformer('x')
        >>> t = t + BagOfTransformers(
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

        if transformers is None:
            self._transformers: List[Transformer] = []
        else:
            self._transformers = transformers

    def insert(self, idx: int, o: Transformer) -> None:
        self._transformers.insert(idx, o)

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", dtypes={self.dtypes}, transformers={self._transformers})'

    def __setitem__(self, idx, o) -> None:
        self._transformers[idx] = o

    def __delitem__(self, idx: Union[int, slice]) -> None:
        del self._transformers[idx]

    def __iter__(self) -> Iterator[Transformer]:
        yield from self._transformers

    def __len__(self) -> int:
        return len(self._transformers)

    def __add__(self, other: 'BagOfTransformers') -> 'BagOfTransformers':
        return BagOfTransformers(name=self.name, transformers=self._transformers + other._transformers)

    def __contains__(self, key: object) -> bool:
        assert isinstance(key, str)
        return True if key in [t.name for t in self._transformers] else False

    def __reversed__(self):
        return reversed(self._transformers)

    def fit(self, df: pd.DataFrame) -> 'BagOfTransformers':
        df = df.copy()  # have to copy because Transformer.transform modifies df

        for transformer in self:
            transformer.fit(df)
            df = transformer.transform(df)

        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self._assert_fitted()

        max_workers = kwargs.pop("max_workers", None)
        if max_workers is None or max_workers > 1:
            try:
                df = self._parallel_transform(df, max_workers, inverse=False, **kwargs)
            except (BrokenProcessPool, OSError):
                logger.warning('Process pool is broken. Running sequentially.')
                self.transform(df, max_workers=0)
        else:
            df = df.copy()
            for transformer in self:
                df = transformer.transform(df, **kwargs)

        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self._assert_fitted()

        max_workers = kwargs.pop("max_workers", None)
        if max_workers is None or max_workers > 1:
            try:
                df = self._parallel_transform(df, max_workers, inverse=True, **kwargs)
            except (BrokenProcessPool, OSError):
                logger.warning('Process pool is broken. Running sequentially.')
                self.inverse_transform(df, max_workers=0)
        else:
            df = df.copy()
            for transformer in reversed(self):
                df = transformer.inverse_transform(df, **kwargs)

        return df

    def _parallel_transform(self, df: pd.DataFrame, max_workers: Optional[int] = None, inverse: bool = False, **kwargs):
        if inverse:
            arguments = ((transformer.inverse_transform,
                          df[list(filter(lambda c: c in df.columns, transformer.out_columns))],
                          transformer.in_columns,
                          kwargs
                          ) for transformer in self)
        else:
            arguments = ((transformer.transform, df[transformer.in_columns], transformer.out_columns, kwargs)
                         for transformer in self)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            columns_transformed = executor.map(self.apply_single_transformation, arguments)
            series = []
            for f in columns_transformed:
                series.append(f)
            return pd.concat(series, axis=1)

        executor.__exit__()

    @staticmethod
    def apply_single_transformation(
        argument: Tuple[Callable, pd.DataFrame, List[str], Dict[str, Any]]
    ) -> pd.DataFrame:
        func, df, out_columns, kwargs = argument
        df = func(df, **kwargs)
        out_columns = list(filter(lambda c: c in df.columns, out_columns))
        return df[out_columns]

    @property
    def in_columns(self) -> List[str]:
        return list(set([column for transformer in self for column in transformer.in_columns]))

    @property
    def out_columns(self) -> List[str]:
        return list(set([column for transformer in self for column in transformer.out_columns]))


class SequentialTransformer(BagOfTransformers, MutableSequence[Transformer]):
    def __init__(self, name: str, transformers: Optional[List[Transformer]] = None, dtypes: Optional[List] = None):
        super().__init__(name, transformers=transformers, dtypes=dtypes)

    def __getitem__(self, idx: Union[int, slice]):
        return self._transformers[idx]

    def __add__(self, other: 'BagOfTransformers') -> 'SequentialTransformer':
        return SequentialTransformer(name=self.name, transformers=self._transformers + other._transformers)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        kwargs['max_workers'] = 1
        return super().transform(df, **kwargs)

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        kwargs['max_workers'] = 1
        return super().inverse_transform(df, **kwargs)
