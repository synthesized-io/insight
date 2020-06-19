import pandas as pd


class BaseMask:
    def __init__(self, column_name, assert_fitted: bool = True):
        self.column_name = column_name
        self.assert_fitted = assert_fitted
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        assert self.column_name in df.columns
        self.fitted = True

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        if self.assert_fitted:
            assert self.fitted, f"Masker for column '{self.column_name}' has not been fitted yet."

        if not inplace:
            df = df.copy()
        assert self.column_name in df.columns
        return df

    def fit_transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df, inplace=inplace)
