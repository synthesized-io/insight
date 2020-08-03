from typing import Any, Dict, Sequence

import tensorflow as tf

from .continuous import ContinuousValue
from .value import Value
from ..module import tensorflow_name_scoped
from ...config import DecomposedContinuousConfig, ContinuousConfig


class DecomposedContinuousValue(Value):

    def __init__(
        self, name: str, config: DecomposedContinuousConfig = DecomposedContinuousConfig()
    ):
        super().__init__(name=name)

        self.weight = config.continuous_weight

        self.low_freq_value = ContinuousValue(
            name=(self.name + '-low-freq'), config=ContinuousConfig(continuous_weight=config.low_freq_weight))
        self.high_freq_value = ContinuousValue(
            name=(self.name + '-high-freq'), config=ContinuousConfig(continuous_weight=config.high_freq_weight))

    def __str__(self) -> str:
        string = super().__str__()
        return string

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            weight=self.weight, low_freq=self.low_freq_value.specification(),
            high_freq=self.high_freq_value.specification()
        )
        return spec

    def learned_input_size(self) -> int:
        return self.low_freq_value.learned_input_size() + self.high_freq_value.learned_input_size()

    def learned_output_size(self) -> int:
        return self.low_freq_value.learned_output_size() + self.high_freq_value.learned_output_size()

    @tensorflow_name_scoped
    def build(self):
        if not self.built:
            self.low_freq_value.build()
            self.high_freq_value.build()

        self.built = True

    @tensorflow_name_scoped
    def unify_inputs(self, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        self.build()
        low = self.low_freq_value.unify_inputs(xs=xs[0:1])
        high = self.high_freq_value.unify_inputs(xs=xs[1:2])
        return tf.concat(values=[low, high], axis=-1)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, **kwargs) -> Sequence[tf.Tensor]:
        y_low_freq, y_high_freq = tf.split(
            value=y,
            num_or_size_splits=[self.low_freq_value.learned_output_size(), self.high_freq_value.learned_output_size()],
            axis=-1
        )

        return self.low_freq_value.output_tensors(y=y_low_freq, **kwargs) + \
            self.high_freq_value.output_tensors(y=y_high_freq, **kwargs)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: Sequence[tf.Tensor]) -> tf.Tensor:

        if len(y.shape) == 2:
            y_low_freq = y[:, 0]
            y_high_freq = y[:, 1]
        elif len(y.shape) == 3:
            y_low_freq = y[:, :, 0]
            y_high_freq = y[:, :, 1]
        else:
            raise NotImplementedError

        loss_low_freq = self.low_freq_value.loss(y=tf.expand_dims(y_low_freq, axis=-1), xs=xs[0:1])
        loss_high_freq = self.high_freq_value.loss(y=tf.expand_dims(y_high_freq, axis=-1), xs=xs[1:2])

        return self.weight * (loss_low_freq + loss_high_freq)
