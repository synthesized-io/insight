from typing import Optional

import pandas as pd

from .base import Transformer, SequentialTransformer
from .categorical import CategoricalTransformer
from ..metadata_new.datetime import Date, get_date_format


class DateTransformer(Transformer):
    """
    Transform datetime64 field to a continuous representation.
    Values are normalised to a timedelta relative to first datetime in data.

    Attributes:
        name: the data frame column to transform.

        date_format: Optional; string representation of date format, eg. '%d/%m/%Y'.

        unit: Optional; unit of timedelta.

        start_date: Optional; when normalising dates, compare relative to this.
    """

    def __init__(self, name: str, date_format: Optional[str] = None, unit: Optional[str] = 'days', start_date: Optional[pd.Timestamp] = None):
        super().__init__(name)
        self.date_format = date_format
        self.unit = unit
        self.start_date = start_date

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", date_format="{self.date_format}", unit="{self.unit}", start_date={self.start_date})'

    def fit(self, x: pd.DataFrame) -> Transformer:

        x = x[self.name]

        if self.date_format is None:
            self.date_format = get_date_format(x)
        if x.dtype.kind != 'M':
            x = pd.to_datetime(x, format=self.date_format)
        if self.start_date is None:
            self.start_date = x.min()

        return super().fit(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if x[self.name].dtype.kind != 'M':
            x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)
        x[self.name] = (x[self.name] - self.start_date).dt.components[self.unit]
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = (pd.to_timedelta(x[self.name], unit=self.unit) + self.start_date).apply(lambda x: x.strftime(self.date_format))
        return x

    @classmethod
    def from_meta(cls, meta: Date) -> 'DateTransformer':
        return cls(meta.name, meta.date_format, start_date=meta.min)


class DateCategoricalTransformer(SequentialTransformer):
    """
    Creates hour, day-of-week, day and month values from a datetime, and
    transforms using CategoricalTransformer.

    Attributes:
        name: the data frame column to transform.

        date_format: Optional; string representation of date format, eg. '%d/%m/%Y'.
    """

    def __init__(self, name: str, date_format: str = None):
        super().__init__(name=name)
        self.date_format = date_format

        self.hour_transform = CategoricalTransformer(f'{self.name}_hour')
        self.dow_transform = CategoricalTransformer(f'{self.name}_dow')
        self.day_transform = CategoricalTransformer(f'{self.name}_day')
        self.month_transform = CategoricalTransformer(f'{self.name}_month')

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", date_format="{self.date_format}")'

    def fit(self, x: pd.DataFrame) -> Transformer:

        x = x[self.name]
        if self.date_format is None:
            self.date_format = get_date_format(x)
        x = self.split_datetime(x)

        for transformer in self.transformers:
            transformer.fit(x)

        return super().fit(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        if x[self.name].dtype.kind != 'M':
            x[self.name] = pd.to_datetime(x[self.name], format=self.date_format)

        categorical_dates = self.split_datetime(x[self.name])
        x[categorical_dates.columns] = categorical_dates

        return super().transform(x)

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x.drop(columns=[f'{self.name}_hour', f'{self.name}_dow',
                        f'{self.name}_day', f'{self.name}_month'], inplace=True)
        return x

    @classmethod
    def from_meta(cls, meta: Date) -> 'DateCategoricalTransformer':
        return cls(meta.name, meta.date_format)

    def split_datetime(self, col: pd.Series) -> pd.DataFrame:
        """Split datetime column into separate hour, dow, day and month fields."""

        if col.dtype.kind != 'M':
            col = pd.to_datetime(col, format=self.date_format)

        return pd.DataFrame({
            f'{col.name}_hour': col.dt.hour,
            f'{col.name}_dow': col.dt.weekday,
            f'{col.name}_day': col.dt.day,
            f'{col.name}_month': col.dt.month
        })
