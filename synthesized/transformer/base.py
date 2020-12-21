from typing import List, Optional, TypeVar, Type, Dict, Union, Iterator, MutableSequence
from abc import abstractmethod
import logging

import pandas as pd
import numpy as np

from .exceptions import NonInvertibleTransformError, TransformerNotFitError

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
    _transformer_registry: Dict[str, Type['Transformer']] = {}

    def __init__(self, name: str, dtypes: Optional[List] = None):
        super().__init__()
        self.name = name
        self.dtypes = dtypes
        self._fitted = False

    def __init_subclass__(cls: Type[TransformerType]):
        super().__init_subclass__()
        Transformer._transformer_registry[cls.__name__] = cls

    def __add__(self, other: 'Transformer') -> 'SequentialTransformer':
        return SequentialTransformer(name=self.name, transformers=[self, other])

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

    def __call__(self, x: pd.DataFrame, inverse=False) -> pd.DataFrame:
        self._assert_fitted()
        if not inverse:
            return self.transform(x)
        else:
            return self.inverse_transform(x)

    def fit(self: TransformerType, x: Union[pd.Series, pd.DataFrame]) -> TransformerType:
        if not self._fitted:
            self._fitted = True
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NonInvertibleTransformError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fit(df).transform(df)
        return df

    def _assert_fitted(self):
        if not self._fitted:
            raise TransformerNotFitError("Transformer not fitted yet, please call 'fit()' before calling transform.")

    @classmethod
    def from_meta(cls: Type[TransformerType], meta) -> TransformerType:
        raise NotImplementedError


class SequentialTransformer(Transformer, MutableSequence[Transformer]):
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

        Alternatively, transformers can be appended (or added)
        >>> t = SequentialTransformer(name='t')
        >>> t.append(CategoricalTransformer('x')
        >>> t = t + SequentialTransformer(
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

    def __getitem__(self, idx: Union[int, slice]):
        return self._transformers[idx]

    def __len__(self) -> int:
        return len(self._transformers)

    def __add__(self, other: Transformer) -> 'SequentialTransformer':
        return SequentialTransformer(name=self.name, transformers=self._transformers + [other])

    def fit(self, df: pd.DataFrame) -> 'SequentialTransformer':
        df = df.copy()  # have to copy because Transformer.transform modifies df
        for transformer in self:
            transformer.fit(df)
            df = transformer.transform(df)
        return super().fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for transformer in self:
            df = transformer.transform(df)
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for transformer in reversed(self):
            df = transformer.inverse_transform(df)
        return df
