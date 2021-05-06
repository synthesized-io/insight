import pandas as pd
from ..base import Transformer
from ..exceptions import NonInvertibleTransformError
from ...metadata import ValueMeta


class NullTransformer(Transformer):
    """
    Transforms by 'nulling out' the values of given column. Deactivates a given column.

    Examples:
        'some_example_value' -> ''

    Attributes:
        name (str) : the data frame column to transform
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}")'

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.loc[:, self.name] = ''
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NonInvertibleTransformError

    @classmethod
    def from_meta(cls, meta: ValueMeta) -> 'NullTransformer':
        return cls(meta.name)
