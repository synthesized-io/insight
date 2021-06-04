from typing import Optional, cast

import pandas as pd

from ..base import Transformer
from ...metadata import Nominal
from ...metadata.value.datetime import DateTime, get_date_format


class DTypeTransformer(Transformer):
    """
    Infer hidden dtype of data and convert it.

    Attributes:
        name (str) : the data frame column to transform.
    """

    def __init__(self, name: str,
                 out_dtype: Optional[str] = None,
                 date_format: Optional[str] = None):
        super().__init__(name=name)
        self.in_dtype: Optional[str] = None
        self.out_dtype: Optional[str] = out_dtype
        self.date_format: Optional[str] = date_format

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, dtypes={self.dtypes}, out_dtype={self.out_dtype})'

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        self.in_dtype = str(df[self.name].dtype)
        if self.out_dtype is not None:
            return super().fit(df)

        column = df[self.name].copy()

        # Try to convert it to numeric
        if column.dtype.kind not in ('i', 'u', 'f') and column.dtype.kind != 'M':
            n_nans = column.isna().sum()
            col_num = pd.to_numeric(column, errors='coerce')
            if col_num.isna().sum() == n_nans:
                column = col_num

        # Try to convert it to date
        if column.dtype.kind == 'O' or column.dtype.kind == 'M':
            n_nans = column.isna().sum()
            try:
                self.date_format = get_date_format(column)
                col_date = pd.to_datetime(column, errors='coerce', format=self.date_format)
            except TypeError:  # Argument 'date_string' has incorrect type (expected str, got numpy.str_)
                col_date = pd.to_datetime(column.astype(str), errors='coerce', format=self.date_format)

            if col_date.isna().sum() == n_nans:
                column = col_date

        self.out_dtype = str(column.dtype)
        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.out_dtype == self.in_dtype:
            return df
        elif self.out_dtype in ('i', 'u', 'f', 'i8', 'u8', 'f8'):
            df[self.name] = pd.to_numeric(df[self.name], errors='coerce')
        else:
            df[self.name] = df[self.name].astype(self.out_dtype, errors='ignore')
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.out_dtype == self.in_dtype:
            return df
        else:
            if self.out_dtype in ('i', 'u', 'i8', 'u8')\
               and self.in_dtype != 'float64':
                if self.out_dtype in ('i', 'i8'):
                    df[self.name] = df[self.name].astype(pd.Int64Dtype(), errors='ignore')
                else:
                    df[self.name] = df[self.name].astype(pd.UInt64Dtype(), errors='ignore')
            elif self.out_dtype in ('M8[ns]', 'datetime64[ns]') and self.date_format is not None:
                df.loc[:, self.name] = pd.to_datetime(df[self.name]).dt.strftime(self.date_format)
            df[self.name] = df[self.name].astype(self.in_dtype, errors='ignore')
        return df

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'DTypeTransformer':
        date_format = None
        if meta.dtype == 'M8[ns]':
            meta = cast(DateTime, meta)
            date_format = meta.date_format
        return cls(meta.name, out_dtype=meta.dtype, date_format=date_format)
