from typing import Optional, Dict, Any
import tensorflow as tf

from .transformation import Transformation
from ..module import tensorflow_name_scoped
from ..util import get_initializer


class BatchNorm(Transformation):
    def __init__(self, input_size: int, name='batch_norm'):
        super(BatchNorm, self).__init__(input_size=input_size, output_size=input_size, name=name)
        self.offset: Optional[tf.Tensor] = None
        self.scale: Optional[tf.Tensor] = None

    @tensorflow_name_scoped
    def build(self, input_shape):
        initializer = get_initializer(initializer='zeros')

        self.offset = self.add_weight(
            name='offset', shape=input_shape, dtype=tf.float32, initializer=initializer,
            trainable=True
        )
        self.scale = self.add_weight(
            name='scale', shape=input_shape, dtype=tf.float32, initializer=initializer,
            trainable=True
        )

        self.built = True

    def call(self, inputs, **kwargs):
        mean, variance = tf.nn.moments(x=inputs, axes=(0,), shift=None, keepdims=False)

        x = tf.nn.batch_normalization(
            x=inputs, mean=mean, variance=variance, offset=self.offset,
            scale=tf.nn.softplus(features=self.scale), variance_epsilon=1e-6
        )
        return x

    def get_variables(self) -> Dict[str, Any]:
        if not self.built:
            raise ValueError(self.name + " hasn't been built yet. ")

        variables = super().get_variables()
        variables.update(
            offset=self.offset.numpy() if self.offset is not None else None,
            scale=self.scale.numpy() if self.scale is not None else None
        )
        return variables

    def set_variables(self, variables: Dict[str, Any]):
        super().set_variables(variables)

        if not self.built:
            self.build(self.input_size)

        if self.offset is not None and self.scale is not None:
            self.offset.assign(variables['offset'])
            self.scale.assign(variables['scale'])
