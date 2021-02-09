from datetime import datetime
from typing import Dict, Optional, Type, TypeVar, cast

import numpy as np
import pandas as pd

from .base import Affine, Domain, Scale
from .exceptions import UnknownDateFormatError

DateType = TypeVar('DateType', bound='Date')


class TimeDelta(Scale[np.timedelta64]):
    class_name: str = 'TimeDelta'
    dtype: str = 'timedelta64[ns]'

    def __init__(
            self, name: str, domain: Optional[Domain[np.timedelta64]] = None, nan_freq: Optional[float] = None,
            min: Optional[np.timedelta64] = None, max: Optional[np.timedelta64] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)


class Date(Affine[np.datetime64]):
    """
    Date meta.

    Date meta describes affine data than can be interpreted as a datetime,
    e.g the string '4/20/2020'.

    Attributes:
        date_format: Optional; string representation of date format, e.g '%d/%m/%Y'.
    """
    class_name: str = 'Date'
    dtype: str = 'datetime64[ns]'

    def __init__(
            self, name: str, domain: Optional[Domain[np.datetime64]] = None, nan_freq: Optional[float] = None,
            min: Optional[np.datetime64] = None, max: Optional[np.datetime64] = None, date_format: Optional[str] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)
        self.date_format = date_format

    def extract(self: DateType, df: pd.DataFrame) -> DateType:
        if self.date_format is None:
            try:
                self.date_format = get_date_format(df[self.name])
            except UnknownDateFormatError:
                self.date_format = None

        df[self.name] = pd.to_datetime(df[self.name], format=self.date_format)
        super().extract(df)  # call super here so we can get max, min from datetime.
        df[self.name] = df[self.name].dt.strftime(self.date_format)

        return self

    def infer_domain(self, df: pd.DataFrame) -> Domain[np.datetime64]:
        return cast(Domain[np.datetime64], pd.to_datetime(df[self.name], format=self.date_format).unique())

    @classmethod
    def unit_meta(cls) -> Type[TimeDelta]:
        return TimeDelta

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
        '%y/%m/%d %H:%M', '%d/%m/%y %H:%M', '%m/%d/%y %H:%M',

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
            sr = sr.apply(lambda x: func(x, date_format))
            parsed_format = date_format
        except ValueError:
            pass
        except TypeError:
            break

    if parsed_format is None:
        raise UnknownDateFormatError("Unable to infer date format.")
    return parsed_format
