from typing import Any, Dict, List, Tuple, Optional

import tensorflow as tf

from . import ContinuousValue
from .value import Value
from ..common.module import tensorflow_name_scoped


class DecomposedContinuousValue(Value):

    def __init__(
        self, name: str, weight: float, identifier: Optional[str],
        # Scenario
        integer: bool = None, float: bool = True, positive: bool = None, nonnegative: bool = None,
        distribution: str = None, distribution_params: Tuple[Any, ...] = None,
        use_quantile_transformation: bool = False,
        transformer_n_quantiles: int = 1000, transformer_noise: Optional[float] = 1e-7,
        low_freq_weight: float = 1., high_freq_weight: float = 1.
    ):
        super().__init__(name=name)

        self.weight = weight
        self.identifier = identifier
        self.low_freq_weight = tf.constant(low_freq_weight, dtype=tf.float32)
        self.high_freq_weight = tf.constant(high_freq_weight, dtype=tf.float32)

        self.weight = weight
        self.integer = integer
        self.float = float
        self.positive = positive
        self.nonnegative = nonnegative
        self.use_quantile_transformation = use_quantile_transformation
        self.transformer_n_quantiles = transformer_n_quantiles
        self.transformer_noise = transformer_noise

        continuous_kwargs: Dict[str, Any] = dict()
        continuous_kwargs['weight'] = weight
        continuous_kwargs['use_quantile_transformation'] = use_quantile_transformation
        continuous_kwargs['transformer_n_quantiles'] = transformer_n_quantiles
        continuous_kwargs['transformer_noise'] = transformer_noise

        self.low_freq_value = ContinuousValue(name=(self.name + '-low-freq'), **continuous_kwargs)
        self.high_freq_value = ContinuousValue(name=(self.name + '-high-freq'), **continuous_kwargs)

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
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        self.build()
        xs[0] = self.low_freq_value.unify_inputs(xs=xs[0:1])
        xs[1] = self.high_freq_value.unify_inputs(xs=xs[1:2])
        return tf.concat(values=xs, axis=-1)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        y_low_freq, y_high_freq = tf.split(value=y,
                                           num_or_size_splits=[
                                               self.low_freq_value.learned_output_size(),
                                               self.high_freq_value.learned_output_size()],
                                           axis=-1)

        return self.low_freq_value.output_tensors(y=y_low_freq, **kwargs) + \
            self.high_freq_value.output_tensors(y=y_high_freq, **kwargs)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:

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

        return self.low_freq_weight * loss_low_freq + self.high_freq_weight * loss_high_freq
