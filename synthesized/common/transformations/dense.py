from math import log

import tensorflow as tf

from .transformation import Transformation
from .. import util
from ..module import tensorflow_name_scoped


class DenseTransformation(Transformation):

    def __init__(
        self, name, input_size, output_size, bias=True, batchnorm=True, activation='relu'
    ):
        super(DenseTransformation, self).__init__(name=name, input_size=input_size, output_size=output_size)

        self.bias = bias
        self.batchnorm = batchnorm
        self.activation = activation

    def specification(self):
        spec = super().specification()
        spec.update(
            bias=self.bias, batchnorm=self.batchnorm, activation=self.activation
        )
        return spec

    @tensorflow_name_scoped
    def build(self, input_shape):
        shape = (self.input_size, self.output_size)
        initializer = util.get_initializer(initializer='glorot-normal')
        self.weight = self.add_weight(
            name='weight', shape=shape, dtype=tf.float32, initializer=initializer, trainable=True
        )

        shape = (self.output_size,)
        initializer = util.get_initializer(initializer='zeros')
        if self.bias:
            self.bias = self.add_weight(
                name='bias', shape=shape, dtype=tf.float32, initializer=initializer, trainable=True
            )
        else:
            self.bias = None

        if self.batchnorm:
            self.offset = self.add_weight(
                name='offset', shape=shape, dtype=tf.float32, initializer=initializer,
                trainable=True
            )
            self.scale = self.add_weight(
                name='scale', shape=shape, dtype=tf.float32, initializer=initializer,
                trainable=True
            )

        self.built = True

    def call(self, inputs, **kwargs):
        x = tf.matmul(
            a=inputs, b=self.weight, transpose_a=False, transpose_b=False, adjoint_a=False,
            adjoint_b=False, a_is_sparse=False, b_is_sparse=False
        )

        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias)

        if self.batchnorm:
            mean, variance = tf.nn.moments(x=x, axes=(0,), shift=None, keepdims=False)
            x = tf.nn.batch_normalization(
                x=x, mean=mean, variance=variance, offset=self.offset,
                scale=tf.nn.softplus(features=self.scale), variance_epsilon=1e-6
            )

        if self.activation == 'none':
            pass
        elif self.activation == 'relu':
            x = tf.nn.relu(features=x)
        elif self.activation == 'softplus':
            # division so that 0.0 is mapped to 1.0, as expected for instance when used for stddev
            x = tf.nn.softplus(features=x) / log(2.0)
        elif self.activation == 'tanh':
            x = tf.tanh(x=x)
        else:
            raise NotImplementedError

        self._output = x

        return x
