from typing import Sequence, Optional, List

import tensorflow as tf

from .categorical import CategoricalValue
from .continuous import ContinuousValue
from ..module import tensorflow_name_scoped
from ...config import CategoricalConfig, ContinuousConfig


class DateValue(ContinuousValue):

    def __init__(
        self, name: str,
        categorical_config: CategoricalConfig = CategoricalConfig(),
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

    def columns(self) -> List[str]:
        return [self.name] + [f"{self.name}_{dt}" for dt in ("hour", "dow", "day", "month")]

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
    def unify_inputs(self, xs: Sequence[tf.Tensor], mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        self.build()
        xs0 = super().unify_inputs(xs=xs[0: 1], mask=mask)
        xs1 = self.hour.unify_inputs(xs=tf.cast(xs[1: 2], dtype=tf.int64))
        xs2 = self.dow.unify_inputs(xs=tf.cast(xs[2: 3], dtype=tf.int64))
        xs3 = self.day.unify_inputs(xs=tf.cast(xs[3: 4], dtype=tf.int64))
        xs4 = self.month.unify_inputs(xs=tf.cast(xs[4: 5], dtype=tf.int64))
        return tf.concat(values=[xs0, xs1, xs2, xs3, xs4], axis=-1)

    def output_tensors(self, y: tf.Tensor, **kwargs) -> Sequence[tf.Tensor]:
        ys = super().output_tensors(y, **kwargs)

        return ys + (tf.zeros_like(ys[0]),) * 4  # hack to match number of columns correctly

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

    def split_outputs(self, outputs):
        return self.convert_tf_to_np_dict({col_name: outputs[self.col_names[0]][i] for i, col_name in enumerate(self.col_names)})
