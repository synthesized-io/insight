from datetime import datetime
from typing import List

import tensorflow as tf

from .categorical import CategoricalValue, CategoricalConfig
from .continuous import ContinuousValue, ContinuousConfig
from ..module import tensorflow_name_scoped


class DateValue(ContinuousValue):

    def __init__(
        self, name: str, categorical_config: CategoricalConfig = CategoricalConfig(),
        continuous_config: ContinuousConfig = ContinuousConfig()
    ):
        super().__init__(name=name, config=continuous_config)

        categorical_config.similarity_based = True
        self.hour = CategoricalValue(
            name=(self.name + '-hour'), num_categories=24, config=categorical_config
        )
        self.dow = CategoricalValue(
            name=(self.name + '-dow'), num_categories=7, config=categorical_config
        )
        self.day = CategoricalValue(
            name=(self.name + '-day'), num_categories=31, config=categorical_config
        )
        self.month = CategoricalValue(
            name=(self.name + '-month'), num_categories=12, config=categorical_config
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            hour=self.hour.specification(), dow=self.dow.specification(),
            day=self.day.specification(), month=self.month.specification()
        )
        return spec

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
