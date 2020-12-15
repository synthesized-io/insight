import pandas as pd

from .base import Transformer
from .exceptions import NonInvertibleTransformError


class DropColumnTransformer(Transformer):
    """
    Drop a column

    Attributes:
        name (str) : the data frame column to transform.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(self.name, axis=1, errors='ignore')

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NonInvertibleTransformError
