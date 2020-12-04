import pandas as pd

from .base import Transformer
from ..metadata_new import Nominal


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

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[f'{self.name}_nan'] = x[self.name].isna().astype(int)
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x.drop(columns=[f'{self.name}_nan'], inplace=True)
        return x

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'NanTransformer':
        return cls(meta.name)
