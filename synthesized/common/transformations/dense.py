from math import log

import tensorflow as tf

from .transformation import Transformation
from ..module import tensorflow_name_scoped
from .. import util


class DenseTransformation(Transformation):

    def __init__(
        self, name, input_size, output_size, bias=True, batchnorm=True, activation='relu',
        weight_decay=0.0
    ):
        super().__init__(name=name, input_size=input_size, output_size=output_size)

        self.bias = bias
        self.batchnorm = batchnorm
        self.activation = activation
        self.weight_decay = weight_decay

    def specification(self):
        spec = super().specification()
        spec.update(
            bias=self.bias, batchnorm=self.batchnorm, activation=self.activation,
            weight_decay=self.weight_decay
        )
        return spec

    def module_initialize(self):
        super().module_initialize()

        shape = (self.input_size, self.output_size)
        initializer = util.get_initializer(initializer='normal')
        regularizer = util.get_regularizer(regularizer='l2', weight=self.weight_decay)
        self.weight = tf.compat.v1.get_variable(
            name='weight', shape=shape, dtype=tf.float32, initializer=initializer,
            regularizer=regularizer, trainable=True
        )

        shape = (self.output_size,)
        initializer = util.get_initializer(initializer='zeros')
        if self.bias:
            self.bias = tf.compat.v1.get_variable(
                name='bias', shape=shape, dtype=tf.float32, initializer=initializer,
                regularizer=regularizer, trainable=True
            )
        else:
            self.bias = None

        if self.batchnorm:
            self.offset = tf.compat.v1.get_variable(
                name='offset', shape=shape, dtype=tf.float32, initializer=initializer,
                trainable=True
            )
            self.scale = tf.compat.v1.get_variable(
                name='scale', shape=shape, dtype=tf.float32, initializer=initializer,
                trainable=True
            )

    @tensorflow_name_scoped
    def transform(self, x):
        x = tf.matmul(
            a=x, b=self.weight, transpose_a=False, transpose_b=False, adjoint_a=False,
            adjoint_b=False, a_is_sparse=False, b_is_sparse=False
        )

        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias)

        if self.batchnorm:
            mean, variance = tf.nn.moments(x=x, axes=(0,), shift=None, keep_dims=False)
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

        # dropout
        return x
