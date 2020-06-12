from datetime import datetime
from typing import Optional

import pandas as pd

from .continuous import ContinuousMeta


class DateMeta(ContinuousMeta):

    def __init__(
        self, name: str, start_date: datetime = None, min_date: datetime = None, keep_monotonic: bool = False
    ):
        super().__init__(name=name, float=True)

        assert start_date is None or min_date is None
        self.start_date = start_date
        self.min_date = min_date
        self.keep_monotonic = keep_monotonic

        self.pd_types = ('M',)
        self.date_format: Optional[str] = None
        self.original_dtype = None

    def specification(self):
        spec = super().specification()
        spec.update(
            keep_monotonic=self.keep_monotonic
        )
        return spec

    def extract(self, df):
        column = df.loc[:, self.name]

        self.original_dtype = type(df.loc[:, self.name].iloc[0])
        if column.dtype.kind != 'M':
            column = self.to_datetime(column)

        if column.is_monotonic and self.keep_monotonic:
            if self.start_date is None:
                self.start_date = column.values[0] - (column.values[1:] - column.values[:-1]).mean()
            elif column[0] < self.start_date:
                raise NotImplementedError
            previous_date = column.values.copy()
            previous_date[1:] = previous_date[:-1]
            previous_date[0] = self.start_date
            column = (column - previous_date).dt.total_seconds() / (24 * 60 * 60)

        else:
            if self.min_date is None:
                self.min_date = column.min()
            elif column.min() != self.min_date:
                raise NotImplementedError
            column = (column - self.min_date).dt.total_seconds() / (24 * 60 * 60)

        super().extract(df=pd.DataFrame.from_dict({self.name: column}))

    def to_datetime(self, col: pd.Series) -> pd.Series:
        formats = (
            '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%d/%m/%Y %H.%M.%S', '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y', '%y-%m-%d', '%m-%d-%y', '%d-%m-%y',
            '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%y/%m/%d', '%m/%d/%y', '%d/%m/%y',
            '%y/%m/%d %H:%M', '%d/%m/%y %H:%M', '%m/%d/%y %H:%M'
        )
        for date_format in formats:
            try:
                def str_to_datetime(in_datetime):
                    return datetime.strptime(in_datetime, date_format)
                col = col.apply(str_to_datetime)
                self.date_format = date_format
                break
            except ValueError:
                pass
            except TypeError:
                break

        if not self.date_format:
            return pd.to_datetime(col)
        return col

    def from_datetime(self, col: pd.Series) -> pd.Series:
        if self.date_format:
            def datetime_to_str(in_datetime):
                return in_datetime.strftime(self.date_format)
            return col.apply(datetime_to_str)
        else:
            return col

    def preprocess(self, df):
        if df.loc[:, self.name].dtype.kind != 'M':
            df.loc[:, self.name] = self.to_datetime(df.loc[:, self.name])

        df[self.name + '-hour'] = df.loc[:, self.name].dt.hour
        df[self.name + '-dow'] = df.loc[:, self.name].dt.weekday
        df[self.name + '-day'] = df.loc[:, self.name].dt.day - 1
        df[self.name + '-month'] = df.loc[:, self.name].dt.month - 1
        if self.min_date is None:
            previous_date = df.loc[:, self.name].copy()
            previous_date[0] = self.start_date
            previous_date[1:] = previous_date[:-1]
            df.loc[:, self.name] = (df.loc[:, self.name] - previous_date).dt.total_seconds() / 86400
        else:
            df.loc[:, self.name] = (df.loc[:, self.name] - self.min_date).dt.total_seconds() / 86400
        return super().preprocess(df=df)

    def postprocess(self, df):
        df = super().postprocess(df=df)
        df.loc[:, self.name] = pd.to_timedelta(arg=df.loc[:, self.name], unit='D')
        if self.start_date is not None:
            df.loc[:, self.name] = self.start_date + df.loc[:, self.name].cumsum(axis=0)
        else:
            df.loc[:, self.name] += self.min_date
        df.loc[:, self.name] = self.from_datetime(df.loc[:, self.name])

        for name in ['-hour', '-dow', '-day', '-month']:
            if self.name + name in df.columns:
                df.drop([self.name + name], axis=1, inplace=True)

        return df
