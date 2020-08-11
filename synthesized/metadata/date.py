from datetime import datetime
from typing import Optional, List

import pandas as pd

from .value_meta import ValueMeta
from .continuous import ContinuousMeta


class DateMeta(ContinuousMeta):

    def __init__(
        self, name: str, start_date: datetime = None, min_date: datetime = None, keep_monotonic: bool = False
    ):
        super().__init__(name=name, is_float=True)

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
        super(ContinuousMeta, self).extract(df=df)  # This is to call ValueMeta.extract()

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

    def learned_input_columns(self) -> List[str]:
        return [self.name] + [f'{self.name}-{postfix}' for postfix in ['hour', 'dow', 'day', 'month']]

    def learned_output_columns(self) -> List[str]:
        return [self.name]

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

        self.set_dtypes(df)
        return df


class TimeIndexMeta(ValueMeta):
    def __init__(
        self, name: str
    ):
        super().__init__(name=name)

        self.freq: Optional[str] = None

    def extract(self, df):
        super().extract(df=df)

        df = df.copy()
        if df.loc[:, self.name].dtype.kind != 'M':
            df.loc[:, self.name] = self.to_datetime(df.loc[:, self.name])

        self.set_index(df)

        self.freq = self.infer_freq(df.unstack([n for n, _ in enumerate(df.index.names) if _ != self.name]))

    def set_index(self, df: pd.DataFrame):
        if df.loc[:, self.name].dtype.kind != 'M':
            df.loc[:, self.name] = self.to_datetime(df.loc[:, self.name])

        if df.index.names == [None]:
            df.set_index(self.name, inplace=True)
        else:
            df.set_index(self.name, inplace=True, append=True)

    def make_index_periodic(self, df: pd.DataFrame):
        other_idx = [n for n in df.index.names if n != self.name]
        df = df.unstack(other_idx).asfreq(
            self.freq, method='ffill'
        ).fillna(
            method='ffill'
        ).stack(other_idx).swaplevel(0, -1, axis=0).sort_index()

        return df

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
        self.set_index(df)
        df = self.make_index_periodic(df)

        df[self.name + '-hour'] = df.index.get_level_values(self.name).hour
        df[self.name + '-dow'] = df.index.get_level_values(self.name).weekday
        df[self.name + '-day'] = df.index.get_level_values(self.name).day - 1
        df[self.name + '-month'] = df.index.get_level_values(self.name).month - 1

        return super().preprocess(df=df)

    def learned_input_columns(self) -> List[str]:
        return [f'{self.name}-{postfix}' for postfix in ['hour', 'dow', 'day', 'month']]

    def learned_output_columns(self) -> List[str]:
        return []

    @staticmethod
    def infer_freq(df: pd.DataFrame):
        freqs = [
            'B', 'D', 'W', 'M', 'SM', 'BM', 'MS', 'SMS', 'BMS', 'Q', 'BQ', 'QS', 'BQS', 'A', 'Y', 'BA', ' BY', 'AS',
            'YS',
            'BAS', 'BYS', 'BH', 'H'
        ]
        best, min = None, None
        for freq in freqs:
            df2 = df.asfreq(freq)
            if len(df2) < len(df):
                continue
            nan_count = sum(df2.iloc[:, 0].isna())
            if min is None or nan_count < min:
                min = nan_count
                best = freq
        return best
