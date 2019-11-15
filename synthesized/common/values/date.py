from typing import List, Optional

import pandas as pd
import tensorflow as tf
from datetime import datetime

from .categorical import CategoricalValue
from .continuous import ContinuousValue
from ..module import tensorflow_name_scoped


class DateValue(ContinuousValue):

    def __init__(
        self, name: str, weight: float, categorical_kwargs: dict, start_date=None, min_date=None
    ):
        super().__init__(name=name, weight=weight)

        assert start_date is None or min_date is None
        self.start_date = start_date
        self.min_date = min_date

        self.pd_types = ('M',)
        self.date_format: Optional[str] = None
        self.pd_cast = (lambda x: pd.to_datetime(x))
        self.original_dtype = None

        self.hour = self.add_module(
            module=CategoricalValue, name=(self.name + '-hour'), categories=24, **categorical_kwargs
        )
        self.dow = self.add_module(
            module=CategoricalValue, name=(self.name + '-dow'), categories=7, **categorical_kwargs
        )
        self.day = self.add_module(
            module=CategoricalValue, name=(self.name + '-day'), categories=31, **categorical_kwargs
        )
        self.month = self.add_module(
            module=CategoricalValue, name=(self.name + '-month'), categories=12, **categorical_kwargs
        )

    def __str__(self):
        string = super().__str__()
        if self.start_date is not None:
            string += '-timedelta'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            hour=self.hour.specification(), dow=self.dow.specification(),
            day=self.day.specification(), month=self.month.specification()
        )
        return spec

    def learned_input_columns(self):
        columns = super().learned_input_columns()
        columns.extend(self.hour.learned_input_columns())
        columns.extend(self.dow.learned_input_columns())
        columns.extend(self.day.learned_input_columns())
        columns.extend(self.month.learned_input_columns())
        return columns

    def learned_input_size(self):
        return super().learned_input_size() + self.hour.learned_input_size() + \
            self.dow.learned_input_size() + self.day.learned_input_size() + \
            self.month.learned_input_size()

    def learned_output_size(self):
        return super().learned_output_size()

    def extract(self, df):
        column = df[self.name]

        self.original_dtype = type(df[self.name].iloc[0])
        if column.dtype.kind != 'M':
            column = self.to_datetime(column)

        if column.is_monotonic:
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

    def preprocess(self, df):
        if df[self.name].dtype.kind != 'M':
            df[self.name] = self.to_datetime(df[self.name])

        df[self.name + '-hour'] = df[self.name].dt.hour
        df[self.name + '-dow'] = df[self.name].dt.weekday
        df[self.name + '-day'] = df[self.name].dt.day - 1
        df[self.name + '-month'] = df[self.name].dt.month - 1
        if self.min_date is None:
            previous_date = df[self.name].copy()
            previous_date[0] = self.start_date
            previous_date[1:] = previous_date[:-1]
            df[self.name] = (df[self.name] - previous_date).dt.total_seconds() / 86400
        else:
            df[self.name] = (df[self.name] - self.min_date).dt.total_seconds() / 86400
        return super().preprocess(df=df)

    def postprocess(self, df):
        df = super().postprocess(df=df)
        df[self.name] = pd.to_timedelta(arg=df[self.name], unit='D')
        if self.start_date is not None:
            df[self.name] = self.start_date + df[self.name].cumsum(axis=0)
        else:
            df[self.name] += self.min_date
        df[self.name] = self.from_datetime(df[self.name])
        return df

    def to_datetime(self, col: pd.Series) -> pd.Series:
        formats = (
            '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%d/%m/%Y %H.%M.%S', '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y', '%y-%m-%d', '%m-%d-%y', '%d-%m-%y',
            '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%y/%m/%d', '%m/%d/%y', '%d/%m/%y',
            '%y/%m/%d %H:%M'
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

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        xs = super().input_tensors()
        xs.extend(self.hour.input_tensors())
        xs.extend(self.dow.input_tensors())
        xs.extend(self.day.input_tensors())
        xs.extend(self.month.input_tensors())
        return xs

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        assert len(xs) == 5
        xs[0] = super().unify_inputs(xs=xs[0: 1])
        xs[1] = self.hour.unify_inputs(xs=xs[1: 2])
        xs[2] = self.dow.unify_inputs(xs=xs[2: 3])
        xs[3] = self.day.unify_inputs(xs=xs[3: 4])
        xs[4] = self.month.unify_inputs(xs=xs[4: 5])
        return tf.concat(values=xs, axis=1)

    # TODO: skip last and assume absolute value
    # def tf_loss(self, x, feed=None):
