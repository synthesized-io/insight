from typing import List, Optional, Union, TypeVar, Type, Dict
from collections import defaultdict
import logging

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

from .exceptions import NonInvertibleTransformError

logger = logging.getLogger(__name__)

TransformerType = TypeVar('TransformerType', bound='Transformer')


# TODO: It looks to me like Transformer and SequentialTransformer should form one single class.
#       Transformer can then be described as:
#       class Transformer(MutableSequence['Transformer'], TransformerMixin)
#       This way, any transformer could be seen as a list containing itself and the __add__ method is simplified:
#       ie. Tf_A + Tf_B ---> [Tf_A] + [Tf_B] = [Tf_A, Tf_B]

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
        self._fitted = False

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, dtypes={self.dtypes})'

    def __add__(self, other: 'Transformer') -> 'SequentialTransformer':
        return SequentialTransformer(name=self.name, transformers=[self, other])

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

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NonInvertibleTransformError

    @classmethod
    def from_meta(cls: Type[TransformerType], meta) -> TransformerType:
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

        self._transformers: List[Transformer] = []
        if transformers is None:
            transformers = []
        self.transformers = transformers

    def __iter__(self):
        yield from self.transformers

    def __getitem__(self, idx):
        return self.transformers[idx]

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", dtypes={self.dtypes}, transformers={self.transformers})'

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
