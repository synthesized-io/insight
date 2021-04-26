from typing import List

import numpy as np
import pandas as pd

from ..base import Transformer
from ...metadata import Nominal


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

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df[self.name].replace([np.inf, -np.inf], np.nan, inplace=True)
        nan = df[self.name].isna()
        df[f'{self.name}_nan'] = nan.astype(int) + 1
        return df

    def inverse_transform(self, df: pd.DataFrame, produce_nans: bool = True, **kwargs) -> pd.DataFrame:
        nan = (df[f'{self.name}_nan'] - 1).astype(bool)  # output produced by value lies in (1, 2)
        if produce_nans:
            df.loc[:, self.name] = df.loc[:, self.name].where(~nan, np.nan)
        df.drop(columns=[f'{self.name}_nan'], inplace=True)
        return df

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'NanTransformer':
        return cls(meta.name)

    @property
    def out_columns(self) -> List[str]:
        return [f'{self.name}', f'{self.name}_nan']
