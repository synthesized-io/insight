from typing import Optional, TypeVar, Dict, Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from .continuous import Integer
from .categorical import String
from ..base import Affine, Scale
from ..exceptions import UnknownDateFormatError

DateType = TypeVar('DateType', bound='Date')


class TimeDelta(Scale[np.timedelta64]):
    dtype = 'm8[ns]'
    precision = np.timedelta64(1, 'ns')

    def __init__(
            self, name: str, categories: Optional[Sequence[np.timedelta64]] = None,
            nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)


class TimeDeltaDay(TimeDelta):
    dtype = 'm8[D]'
    precision = np.timedelta64(1, 'D')


class Date(Affine[np.datetime64]):
    """
    Date meta.

    Date meta describes affine data than can be interpreted as a datetime,
    e.g the string '4/20/2020'.

    Attributes:
        date_format: Optional; string representation of date format, e.g '%d/%m/%Y'.
    """
    dtype = 'M8[D]'

    def __init__(
            self, name: str, categories: Optional[Sequence[np.datetime64]] = None, nan_freq: Optional[float] = None,
            date_format: Optional[str] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)
        self.date_format = date_format
        self.children = [
            String(name + '_dow'), Integer(name + '_day'), Integer(name + '_month'), Integer(name + '_year')
        ]

        self._unit_meta = TimeDeltaDay(self.name + '_unit')

    def extract(self: DateType, df: pd.DataFrame):
        if self.date_format is None:
            try:
                self.date_format = get_date_format(df[self.name])
            except UnknownDateFormatError:
                self.date_format = None

        df[self.name] = pd.to_datetime(df[self.name], format=self.date_format)

        sub_df = pd.DataFrame({
            self.name: df[self.name],
            self.name + '_dow': df[self.name].dt.day_name(),
            self.name + '_day': df[self.name].dt.day,
            self.name + '_month': df[self.name].dt.month,
            self.name + '_year': df[self.name].dt.year,
        })

        super().extract(sub_df)  # call super here so we can get max, min from datetime.

        return self

    @property
    def unit_meta(self) -> TimeDeltaDay:
        return self._unit_meta

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "date_format": self.date_format
        })

        return d


def get_date_format(sr: pd.Series) -> str:
    """
    Infer the date format for a series of dates.

    Returns:
        date format string, e.g "%d/%m/%Y.

    Raises:
        UnknownDateFormatError: date format cannot be inferred.
    """
    formats = (
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%d/%m/%Y %H.%M.%S', '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y', '%y-%m-%d', '%m-%d-%y', '%d-%m-%y',
        '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%y/%m/%d', '%m/%d/%y', '%d/%m/%y',
        '%y/%m/%d %H:%M', '%d/%m/%y %H:%M', '%m/%d/%y %H:%M'
    )
    parsed_format = None
    if sr.dtype.kind == 'M':
        func = datetime.strftime
    else:
        func = datetime.strptime  # type: ignore
    for date_format in formats:
        try:
            sr = sr.apply(lambda x: func(x, date_format))
            parsed_format = date_format
        except ValueError:
            pass
        except TypeError:
            break

    if parsed_format is None:
        raise UnknownDateFormatError("Unable to infer date format.")
    return parsed_format
