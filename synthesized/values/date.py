from datetime import datetime
from typing import List, Optional

from dataclasses import dataclass, fields
import tensorflow as tf

from .categorical import CategoricalValue
from .continuous import ContinuousValue
from ..common.module import tensorflow_name_scoped


@dataclass
class DateConfig:
    keep_monotonic_dates: bool = False

    @property
    def date_config(self):
        return DateConfig(**{f.name: self.__getattribute__(f.name) for f in fields(DateConfig)})


class DateValue(ContinuousValue):

    def __init__(
        self, name: str, categorical_kwargs: dict, continuous_kwargs: dict,
        start_date: datetime = None, min_date: datetime = None, keep_monotonic: bool = False
    ):
        super().__init__(name=name, **continuous_kwargs)

        assert start_date is None or min_date is None
        self.start_date = start_date
        self.min_date = min_date
        self.keep_monotonic = keep_monotonic

        self.pd_types = ('M',)
        self.date_format: Optional[str] = None
        self.original_dtype = None

        categorical_kwargs['similarity_based'] = True
        self.hour = CategoricalValue(
            name=(self.name + '-hour'), categories=list(range(24)), **categorical_kwargs
        )
        self.dow = CategoricalValue(
            name=(self.name + '-dow'), categories=list(range(7)), **categorical_kwargs
        )
        self.day = CategoricalValue(
            name=(self.name + '-day'), categories=list(range(31)), **categorical_kwargs
        )
        self.month = CategoricalValue(
            name=(self.name + '-month'), categories=list(range(12)), **categorical_kwargs
        )

    def __str__(self):
        string = super().__str__()
        if self.start_date is not None:
            string += '-timedelta'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            keep_monotonic=self.keep_monotonic,
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

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        self.build()
        xs[0] = super().unify_inputs(xs=xs[0: 1])
        xs[1] = self.hour.unify_inputs(xs=tf.cast(xs[1: 2], dtype=tf.int64))
        xs[2] = self.dow.unify_inputs(xs=tf.cast(xs[2: 3], dtype=tf.int64))
        xs[3] = self.day.unify_inputs(xs=tf.cast(xs[3: 4], dtype=tf.int64))
        xs[4] = self.month.unify_inputs(xs=tf.cast(xs[4: 5], dtype=tf.int64))
        return tf.concat(values=xs, axis=-1)

    # TODO: skip last and assume absolute value
    # def tf_loss(self, x, feed=None):

    @tensorflow_name_scoped
    def build(self):
        if not self.built:
            self.hour.build()
            self.dow.build()
            self.day.build()
            self.month.build()
            self.built = True
