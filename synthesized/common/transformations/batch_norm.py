import tensorflow as tf

from .transformation import Transformation
from ..module import tensorflow_name_scoped
from ..util import get_initializer


class BatchNorm(Transformation):
    def __init__(self, input_size: int, name='batch_norm'):
        super(BatchNorm, self).__init__(input_size=input_size, output_size=input_size, name=name)

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
