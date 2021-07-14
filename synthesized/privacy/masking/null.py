import pandas as pd

from synthesized.metadata import ValueMeta
from synthesized.transformer.base import Transformer
from synthesized.transformer.exceptions import NonInvertibleTransformError


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
        """Fits the given dataframe to the transformer

        Args:
            df: Dataset to fit

        Returns:
            self
        """
        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transforms the given dataframe using fitted transformer

        Args:
            df: Dataset to transform

        Returns:
            Transformed dataset
        """
        df.loc[:, self.name] = ''
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NonInvertibleTransformError

    @classmethod
    def from_meta(cls, meta: ValueMeta) -> 'NullTransformer':
        return cls(meta.name)
