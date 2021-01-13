import numpy as np
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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        nan = df[self.name].isna()
        df[f'{self.name}_nan'] = nan.astype(int)
        return df

    def inverse_transform(self, df: pd.DataFrame, produce_nans: bool = True) -> pd.DataFrame:
        print("self.name", self.name)
        print("df.columns", df.columns)
        nan = df[f'{self.name}_nan'].astype(bool)
        if produce_nans:
            df[self.name] = df[self.name].where(~nan, np.nan)
        df.drop(columns=[f'{self.name}_nan'], inplace=True)
        return df

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'NanTransformer':
        return cls(meta.name)
