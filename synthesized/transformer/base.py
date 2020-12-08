from typing import List, Optional, TypeVar, Type, Dict, Union, MutableSequence
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TransformerType = TypeVar('TransformerType', bound='Transformer')

_transformer_registry: Dict[str, Type['Transformer']] = {}


class Transformer(MutableSequence['Transformer']):
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

    def __init__(self, name: str, transformers: Optional[List['Transformer']] = None, dtypes: Optional[List] = None):
        super().__init__()
        self.name = name
        self.dtypes = dtypes
        self._fitted = False

        if transformers is None:
            self._transformers: List[Transformer] = []
        else:
            self._transformers = transformers

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _transformer_registry[cls.__name__] = cls

    def insert(self, idx: int, o: 'Transformer') -> None:
        """__getitem__, __setitem__, __delitem__, __iter__, and __len__"""
        self._transformers.insert(idx, o)

    def __setitem__(self, idx, o) -> None:
        self._transformers[idx] = o

    def __delitem__(self, idx) -> None:
        del self._transformers[idx]

    def __iter__(self):
        yield from self._transformers

    def __getitem__(self, idx):
        return self._transformers[idx]

    def __len__(self):
        return len(self._transformers)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, dtypes={self.dtypes})'

    def __add__(self: TransformerType, other: 'Transformer') -> 'TransformerType':
        self.append(other)
        return self

    def __eq__(self, other):
        def attrs(x):
            return dict(filter(lambda x: not x[0].startswith('_'), x.__dict__.items()))
        try:
            np.testing.assert_equal(attrs(self), attrs(other))
            return True
        except AssertionError:
            return False

    def __call__(self, x: pd.DataFrame, inverse=False) -> pd.DataFrame:
        if not inverse:
            return self.transform(x)
        else:
            return self.inverse_transform(x)

    def fit(self: TransformerType, x: Union[pd.Series, pd.DataFrame]) -> TransformerType:
        if not self._fitted:
            self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Overriding methods should call super().transform(df=df) at the end of the function."""
        for transformer in self:
            df = transformer.fit_transform(df)
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Overriding methods should call super().transform(df=df) at the start of the function."""
        for transformer in reversed(self):
            df = transformer.inverse_transform(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fit(df).transform(df)
        return df

    @classmethod
    def from_meta(cls: Type[TransformerType], meta) -> TransformerType:
        raise NotImplementedError
