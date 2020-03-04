from math import log

import tensorflow as tf

from .transformation import Transformation
from .batch_norm import BatchNorm
from ..util import get_initializer
from ..module import tensorflow_name_scoped


class DenseTransformation(Transformation):

    def __init__(
        self, name, input_size, output_size, bias=True, batchnorm=True, activation='relu'
    ):
        super(DenseTransformation, self).__init__(name=name, input_size=input_size, output_size=output_size)

        self.bias = bias
        self.activation = activation
        self.batchnorm = BatchNorm(input_size=output_size) if batchnorm else False

    def specification(self):
        spec = super().specification()
        spec.update(
            bias=self.bias, batchnorm=self.batchnorm, activation=self.activation
        )
        return spec

    @tensorflow_name_scoped
    def build(self, input_shape):
        shape = (self.input_size, self.output_size)
        initializer = get_initializer(initializer='glorot-normal')
        self.weight = self.add_weight(
            name='weight', shape=shape, dtype=tf.float32, initializer=initializer, trainable=True
        )
        self.add_regularization_weights(self.weight)

        shape = (self.output_size,)
        initializer = get_initializer(initializer='zeros')
        if self.bias:
            self.bias = self.add_weight(
                name='bias', shape=shape, dtype=tf.float32, initializer=initializer, trainable=True
            )
            self.add_regularization_weights(self.bias)
        else:
            self.bias = None

        if self.batchnorm:
            self.batchnorm.build(input_shape=shape)

        self.built = True

    def call(self, inputs, **kwargs):
        x = tf.matmul(
            a=inputs, b=self.weight, transpose_a=False, transpose_b=False, adjoint_a=False,
            adjoint_b=False, a_is_sparse=False, b_is_sparse=False
        )

        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias)

        if self.batchnorm:
            x = self.batchnorm(x)

        if self.activation == 'none':
            pass
        elif self.activation == 'relu':
            x = tf.nn.relu(features=x)
        elif self.activation == 'leaky_relu':
            x = tf.nn.leaky_relu(features=x)
        elif self.activation == 'softplus':
            # division so that 0.0 is mapped to 1.0, as expected for instance when used for stddev
            x = tf.nn.softplus(features=x) / log(2.0)
        elif self.activation == 'tanh':
            x = tf.tanh(x=x)
        else:
            raise NotImplementedError

        tf.summary.histogram(name='weight', data=self.weight)
        tf.summary.histogram(name='bias', data=self.bias)

        self._output = x

        return x
