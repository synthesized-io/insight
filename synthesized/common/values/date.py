from typing import Sequence

import tensorflow as tf

from ...config import CategoricalConfig, ContinuousConfig
from ..module import tensorflow_name_scoped
from .categorical import CategoricalValue
from .continuous import ContinuousValue


class DateValue(ContinuousValue):

    def __init__(
        self, name: str, categorical_config: CategoricalConfig = CategoricalConfig(),
        continuous_config: ContinuousConfig = ContinuousConfig()
    ):
        super().__init__(name=name, config=continuous_config)

        # Number of categories includes NaN category
        self.hour = CategoricalValue(
            name=(self.name + '-hour'), num_categories=25, similarity_based=True, config=categorical_config
        )
        self.dow = CategoricalValue(
            name=(self.name + '-dow'), num_categories=8, similarity_based=True, config=categorical_config
        )
        self.day = CategoricalValue(
            name=(self.name + '-day'), num_categories=32, similarity_based=True, config=categorical_config
        )
        self.month = CategoricalValue(
            name=(self.name + '-month'), num_categories=13, similarity_based=True, config=categorical_config
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
    def unify_inputs(self, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        self.build()
        xs0 = super().unify_inputs(xs=xs[0: 1])
        xs1 = self.hour.unify_inputs(xs=tf.cast(xs[1: 2], dtype=tf.int64))
        xs2 = self.dow.unify_inputs(xs=tf.cast(xs[2: 3], dtype=tf.int64))
        xs3 = self.day.unify_inputs(xs=tf.cast(xs[3: 4], dtype=tf.int64))
        xs4 = self.month.unify_inputs(xs=tf.cast(xs[4: 5], dtype=tf.int64))
        return tf.concat(values=[xs0, xs1, xs2, xs3, xs4], axis=-1)

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

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: Sequence[tf.Tensor], mask: tf.Tensor = None) -> tf.Tensor:
        xs = xs[0:1]
        loss = super().loss(y=y, xs=xs, mask=mask)
        return loss
