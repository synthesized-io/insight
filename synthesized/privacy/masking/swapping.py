from typing import Optional

import numpy as np
import pandas as pd

from .base_mask import BaseMask


class SwappingMask(BaseMask):
    def __init__(self, column_name: str, uniform: bool = False):
        super(SwappingMask, self).__init__(column_name)
        self.uniform = uniform
        self.categories: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame):

        self.categories = df.loc[:, self.column_name].value_counts(normalize=True, sort=True, dropna=False)
        if self.uniform:
            p = 1 / len(self.categories)
            self.categories.apply(lambda x: p)

        super(SwappingMask, self).fit(df)

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        df = super(SwappingMask, self).transform(df, inplace)

        assert self.categories is not None
        df.loc[:, self.column_name] = np.random.choice(
            a=self.categories.index, size=len(df), p=self.categories.values
        )
        return df
