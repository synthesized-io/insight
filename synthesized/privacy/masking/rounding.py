import numpy as np
import pandas as pd

from .base_mask import BaseMask


class RoundingMask(BaseMask):
    def __init__(self, column_name, n_bins: int = 20):
        super(RoundingMask, self).__init__(column_name)
        self.n_bins = n_bins
        self.bins: np.array = None

    def fit(self, df):
        column = pd.to_numeric(df[self.column_name], errors='coerce')
        if pd.to_numeric(df[self.column_name], errors='coerce').isna().all():
            raise ValueError(f"Can apply masking technique 'rounding' to column '{self.column_name}' "
                             f"as it doesn't contain numerical values.")

        _, self.bins = pd.qcut(column, q=self.n_bins, retbins=True, duplicates='drop')

        super(RoundingMask, self).fit(df)

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        df = super(RoundingMask, self).transform(df, inplace)
        df.loc[:, self.column_name] = pd.to_numeric(df.loc[:, self.column_name], errors='coerce')
        df.loc[:, self.column_name] = pd.cut(df.loc[:, self.column_name], bins=self.bins, include_lowest=True,
                                             duplicates='drop').astype(str)
        return df
