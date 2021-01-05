import pandas as pd

from ..metadata_new import Nominal
from .base import Transformer


class NanTransformer(Transformer):
    """
    Creates a Boolean field to indicate the presence of a NaN value in an affine field.

    Attributes:
        name: the data frame column to transform.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}")'

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f'{self.name}_nan'] = df[self.name].isna().astype(int)
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=[f'{self.name}_nan'], inplace=True)
        return df

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'NanTransformer':
        return cls(meta.name)
