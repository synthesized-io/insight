from typing import List, Optional

import numpy as np
import pandas as pd

from .categorical import CategoricalTransformer
from .quantile import QuantileTransformer
from ..base import SequentialTransformer, Transformer
from ...config import DateTransformerConfig
from ...metadata.value.datetime import Affine, get_date_format


class DateTransformer(SequentialTransformer):

    def __init__(
            self, name: str, start_date: Optional[pd.Timestamp] = None, config: Optional[DateTransformerConfig] = None
    ):
        config = DateTransformerConfig() if config is None else config
        transformers = [
            DateCategoricalTransformer(name=name),
            DateToNumericTransformer(name=name, unit=config.unit, start_date=start_date),
            QuantileTransformer(name=name, config=config.quantile_transformer_config)
        ]
        super().__init__(name, transformers=transformers)

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = super().inverse_transform(df)
        df.loc[:, self.name] = pd.to_datetime(df[self.name]).dt.strftime(self[0].date_format)
        return df

    @classmethod
    def from_meta(cls, meta: Affine[np.datetime64], config: Optional[DateTransformerConfig] = None) -> 'DateTransformer':
        return cls(meta.name, start_date=meta.min, config=config)


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

    def __init__(self, name: str, unit: Optional[str] = 'days', start_date: Optional[pd.Timestamp] = None):
        super().__init__(name)
        self.date_format: Optional[str] = None
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
            assert self.date_format is not None
            sr = pd.to_datetime(sr, format=self.date_format, errors='coerce')

        if self.start_date is None:
            self.start_date = sr.min()

        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df[self.name].dtype.kind != 'M':
            df.loc[:, self.name] = pd.to_datetime(df.loc[:, self.name], format=self.date_format, errors='coerce')
        df.loc[:, self.name] = (df.loc[:, self.name] - self.start_date).dt.total_seconds() / 86400
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.loc[:, self.name] = pd.to_timedelta(df[self.name], unit=self.unit) + self.start_date
        return df

    @classmethod
    def from_meta(cls, meta: Affine[np.datetime64]) -> 'DateToNumericTransformer':
        return cls(meta.name, start_date=meta.min)


class DateCategoricalTransformer(Transformer):
    """
    Creates hour, day-of-week, day and month values from a datetime, and
    transforms using CategoricalTransformer.

    Attributes:
        name: the data frame column to transform.

        date_format: Optional; string representation of date format, eg. '%d/%m/%Y'.
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        self.date_format: Optional[str] = None

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
            df.loc[:, self.name] = pd.to_datetime(df.loc[:, self.name], format=self.date_format, errors='coerce')

        categorical_dates = self.split_datetime(df[self.name])
        df[categorical_dates.columns] = categorical_dates

        return self._transformer.transform(df)

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.drop(columns=[f'{self.name}_hour', f'{self.name}_dow',
                         f'{self.name}_day', f'{self.name}_month'], errors='ignore', inplace=True)
        return df

    @classmethod
    def from_meta(cls, meta: Affine[np.datetime64]) -> 'DateCategoricalTransformer':
        return cls(meta.name)

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