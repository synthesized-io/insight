import pandas as pd

from .base_mask import BaseMask


class NullMask(BaseMask):
    def __init__(self, column_name: str):
        super(NullMask, self).__init__(column_name, assert_fitted=False)

    def fit(self, df: pd.DataFrame):
        super(NullMask, self).fit(df)

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        df = super(NullMask, self).transform(df, inplace)
        df.loc[:, self.column_name] = ''
        return df
