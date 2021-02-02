from typing import List, Optional

import pandas as pd

from .categorical import CategoricalTransformer
from .quantile import QuantileTransformer
from ..base import SequentialTransformer, Transformer
from ...metadata_new.value.datetime import Date, get_date_format


class DateTransformer(SequentialTransformer):

    def __init__(self, name: str, date_format: Optional[str] = None, unit: Optional[str] = 'days',
                 start_date: Optional[pd.Timestamp] = None, n_quantiles: int = 1000,
                 output_distribution: str = 'normal', noise: Optional[float] = 1e-7):

        transformers = [
            DateCategoricalTransformer(name=name, date_format=date_format),
            DateToNumericTransformer(name=name, date_format=date_format, unit=unit, start_date=start_date),
            QuantileTransformer(name=name, n_quantiles=n_quantiles, output_distribution=output_distribution,
                                noise=noise)
        ]
        super().__init__(name, transformers=transformers)

    @classmethod
    def from_meta(cls, meta: Date) -> 'DateTransformer':
        return cls(meta.name, date_format=meta.date_format, start_date=meta.min)


class DateToNumericTransformer(Transformer):
    """
    Transform datetime64 field to a continuous representation.
    Values are normalised to a timedelta relative to first datetime in data.

    Attributes:
        name: the data frame column to transform.

        date_format: Optional; string representation of date format, eg. '%d/%m/%Y'.

        unit: Optional; unit of timedelta.

        start_date: Optional; when normalising dates, compare relative to this.
    """

    def __init__(self, name: str, date_format: Optional[str] = None, unit: Optional[str] = 'days',
                 start_date: Optional[pd.Timestamp] = None):
        super().__init__(name)
        self.date_format = date_format
        self.unit = unit
        self.start_date = start_date

    def __repr__(self):
        return (f'{self.__class__.__name__}(name="{self.name}", date_format="{self.date_format}", unit="{self.unit}", '
                f'start_date={self.start_date})')

    def fit(self, df: pd.DataFrame) -> 'DateToNumericTransformer':
        sr = df[self.name]

        if self.date_format is None:
            self.date_format = get_date_format(sr)
        if sr.dtype.kind != 'M':
            sr = pd.to_datetime(sr, format=self.date_format)
        if self.start_date is None:
            self.start_date = sr.min()

        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df[self.name].dtype.kind != 'M':
            df[self.name] = pd.to_datetime(df[self.name], format=self.date_format)
        df[self.name] = (df[self.name] - self.start_date).dt.components[self.unit]
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df[self.name] = (pd.to_timedelta(df[self.name], unit=self.unit) + self.start_date).apply(lambda x: x.strftime(self.date_format))
        return df

    @classmethod
    def from_meta(cls, meta: Date) -> 'DateToNumericTransformer':
        return cls(meta.name, meta.date_format, start_date=meta.min)


class DateCategoricalTransformer(Transformer):
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

        hour_transform = CategoricalTransformer(f'{self.name}_hour')
        dow_transform = CategoricalTransformer(f'{self.name}_dow')
        day_transform = CategoricalTransformer(f'{self.name}_day')
        month_transform = CategoricalTransformer(f'{self.name}_month')

        self._transformer = SequentialTransformer(
            name='date',
            transformers=[hour_transform, dow_transform, day_transform, month_transform]
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", date_format="{self.date_format}")'

    def fit(self, df: pd.DataFrame) -> 'DateCategoricalTransformer':

        sr = df[self.name]
        if self.date_format is None:
            self.date_format = get_date_format(sr)
        df = self.split_datetime(sr)

        self._transformer.fit(df)
        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if df[self.name].dtype.kind != 'M':
            df[self.name] = pd.to_datetime(df[self.name], format=self.date_format)

        categorical_dates = self.split_datetime(df[self.name])
        df[categorical_dates.columns] = categorical_dates

        return self._transformer.transform(df)

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.drop(columns=[f'{self.name}_hour', f'{self.name}_dow',
                         f'{self.name}_day', f'{self.name}_month'], errors='ignore', inplace=True)
        return df

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

    @property
    def out_columns(self) -> List[str]:
        return [self.name, f'{self.name}_hour', f'{self.name}_dow', f'{self.name}_day', f'{self.name}_month']
