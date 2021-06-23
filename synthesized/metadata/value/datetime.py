from datetime import datetime
from typing import Dict, Optional, Sequence, TypeVar, cast

import numpy as np
import pandas as pd

from .categorical import String
from .continuous import Integer
from ..base import Affine, Scale, ValueMeta
from ..exceptions import MetaNotExtractedError, UnknownDateFormatError

DateType = TypeVar('DateType', bound='DateTime')


class TimeDelta(Scale[np.timedelta64]):
    dtype = 'm8[ns]'
    _precision = np.timedelta64(1, 'ns')

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[np.timedelta64]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, unit_meta: Optional['TimeDelta'] = None
    ):
        super().__init__(
            name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows,
            unit_meta=unit_meta
        )


class TimeDeltaDay(TimeDelta):
    dtype = 'm8[D]'
    _precision = np.timedelta64(1, 'D')


class DateTime(Affine[np.datetime64]):
    """
    DateTime meta.

    DateTime meta describes affine data than can be interpreted as a datetime,
    e.g the string '4/20/2020'.

    Attributes:
        date_format: Optional; string representation of date format, e.g '%d/%m/%Y'.
    """
    dtype = 'M8[ns]'

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[np.datetime64]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None,
            unit_meta: Optional[TimeDelta] = None, date_format: Optional[str] = None
    ):
        children = [
            String(name + '_dow'), Integer(name + '_day'), Integer(name + '_month'), Integer(name + '_year')
        ] if children is None else children

        super().__init__(
            name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows,
            unit_meta=unit_meta
        )
        self.date_format = date_format

    def extract(self: DateType, df: pd.DataFrame):
        if self.date_format is None:
            try:
                self.date_format = get_date_format(df[self.name])
            except UnknownDateFormatError:
                self.date_format = None

        sub_df = pd.DataFrame({
            self.name: pd.to_datetime(df[self.name], format=self.date_format, errors='coerce'),
        })

        self.categories = list(sub_df[self.name].dropna().unique())

        super().extract(sub_df)  # call super here so we can get max, min from datetime.

        return self

    def update_meta(self: DateType, df: pd.DataFrame):
        if self.date_format is None:
            try:
                self.date_format = get_date_format(df[self.name])
            except UnknownDateFormatError:
                self.date_format = None

        sub_df = pd.DataFrame({
            self.name: pd.to_datetime(df[self.name], format=self.date_format, errors='coerce'),
        })

        categories = list(sub_df[self.name].dropna().unique())
        self.categories = list(set([y for x in [categories, self.categories] for y in x]))
        super().update_meta(sub_df)

        return self

    def _create_unit_meta(self) -> TimeDelta:
        return TimeDelta(f"{self.name}'", unit_meta=TimeDelta(f"{self.name}''"))

    def convert_df_for_children(self, df: pd.DataFrame):

        sr_dt = df[self.name]

        df[self.name + '_dow'] = sr_dt.dt.day_name()
        df[self.name + '_second'] = sr_dt.dt.second
        df[self.name + '_minute'] = sr_dt.dt.minute
        df[self.name + '_hour'] = sr_dt.dt.hour
        df[self.name + '_day'] = sr_dt.dt.day
        df[self.name + '_month'] = sr_dt.dt.month
        df[self.name + '_year'] = sr_dt.dt.year
        df.drop(columns=self.name, inplace=True)

    def revert_df_from_children(self, df):

        df.loc[:, self.name] = pd.to_datetime(pd.DataFrame({
            'year': df.loc[:, self.name + '_year'],
            'month': df.loc[:, self.name + '_month'],
            'day': df.loc[:, self.name + '_day'],
            'hour': df.loc[:, self.name + '_hour'],
            'minute': df.loc[:, self.name + '_minute'],
            'second': df.loc[:, self.name + '_second'],
        }))

        df.drop(
            columns=[
                self.name + '_dow', self.name + '_day', self.name + '_month', self.name + '_year',
                self.name + '_hour', self.name + '_minute', self.name + '_second'
            ],
            inplace=True
        )

    @property
    def unit_meta(self) -> TimeDelta:
        if self._unit_meta is None:
            raise MetaNotExtractedError(f'Meta {self.name} not extracted.')
        return cast(TimeDelta, self._unit_meta)

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "date_format": self.date_format
        })

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'DateTime':

        name = cast(str, d["name"])
        extracted = cast(bool, d["extracted"])
        num_rows = cast(Optional[int], d["num_rows"])
        nan_freq = cast(Optional[float], d["nan_freq"])
        categories = cast(Optional[Sequence[np.datetime64]], d["categories"])
        children: Optional[Sequence[ValueMeta]] = ValueMeta.children_from_dict(d)
        d_unit = cast(Dict[str, object], d["unit_meta"]) if d["unit_meta"] is not None else None
        date_format = cast(str, d["date_format"])
        unit_meta = TimeDelta.from_name_and_dict(cast(str, d_unit["class_name"]), d_unit) if d_unit is not None else None

        meta = cls(
            name=name, children=children, num_rows=num_rows, nan_freq=nan_freq, categories=categories,
            unit_meta=unit_meta, date_format=date_format
        )
        meta._extracted = extracted

        return meta


def get_date_format(sr: pd.Series) -> str:
    """
    Infer the date format for a series of dates.

    Returns:
        date format string, e.g "%d/%m/%Y.

    Raises:
        UnknownDateFormatError: date format cannot be inferred.
    """
    sr = sr.dropna()

    # Attempt to parse the smaller formats (eg. Y, m, d) before the larger ones (eg. Y, m, d, H, M, S)
    formats = (
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%d/%m/%Y %H.%M.%S', '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y', '%y-%m-%d', '%m-%d-%y', '%d-%m-%y',
        '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%y/%m/%d', '%m/%d/%y', '%d/%m/%y',
        '%y/%m/%d %H:%M', '%d/%m/%y %H:%M', '%m/%d/%y %H:%M', '%d/%m/%Y %H:%M',

        '%Y-%b-%d %H:%M:%S', '%Y-%b-%dT%H:%M:%SZ', '%d/%b/%Y %H.%M.%S', '%d/%b/%Y %H:%M:%S',
        '%Y-%b-%d', '%b-%d-%Y', '%d-%b-%Y', '%y-%b-%d', '%b-%d-%y', '%d-%b-%y',
        '%Y/%b/%d', '%b/%d/%Y', '%d/%b/%Y', '%y/%b/%d', '%b/%d/%y', '%d/%b/%y',
        '%y/%b/%d %H:%M', '%d/%b/%y %H:%M', '%b/%d/%y %H:%M',
    )
    parsed_format = None
    if sr.dtype.kind == 'M':
        func = datetime.strftime
    else:
        func = datetime.strptime  # type: ignore
    for date_format in formats:
        try:
            sr = sr.apply(lambda x, date_format=date_format: func(x, date_format))
            parsed_format = date_format
        except ValueError:
            pass
        except TypeError:
            break

    if parsed_format is None:
        raise UnknownDateFormatError("Unable to infer date format.")
    return parsed_format
