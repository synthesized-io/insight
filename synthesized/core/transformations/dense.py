import tensorflow as tf

from .transformation import Transformation
from .. import util


class DenseTransformation(Transformation):

    def __init__(self, name, input_size, output_size, bias=True, batchnorm=True, activation='relu'):
        super().__init__(name=name, input_size=input_size, output_size=output_size)
        self.bias = bias
        self.batchnorm = batchnorm
        self.activation = activation

    def specification(self):
        spec = super().specification()
        spec.update(bias=self.bias, batchnorm=self.batchnorm, activation=self.activation)
        return spec

    def tf_initialize(self):
        super().tf_initialize()
        self.weight = tf.get_variable(
            name='weight', shape=(self.input_size, self.output_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        if self.bias:
            self.bias = tf.get_variable(
                name='bias', shape=(self.output_size,), dtype=tf.float32,
                initializer=util.initializers['zero'], regularizer=util.regularizers['l2'],
                trainable=True, collections=None, caching_device=None, partitioner=None,
                validate_shape=True, use_resource=None, custom_getter=None
            )
        else:
            self.bias = None
        if self.batchnorm:
            self.offset = tf.get_variable(
                name='offset', shape=(self.output_size,), dtype=tf.float32,
                initializer=util.initializers['zero'], regularizer=util.regularizers['l2'],
                trainable=True, collections=None, caching_device=None, partitioner=None,
                validate_shape=True, use_resource=None, custom_getter=None
            )
            self.scale = tf.get_variable(
                name='scale', shape=(self.output_size,), dtype=tf.float32,
                initializer=util.initializers['one'], regularizer=util.regularizers['l2'],
                trainable=True, collections=None, caching_device=None, partitioner=None,
                validate_shape=True, use_resource=None, custom_getter=None
            )

    def tf_transform(self, x):
        x = tf.matmul(
            a=x, b=self.weight, transpose_a=False, transpose_b=False, adjoint_a=False,
            adjoint_b=False, a_is_sparse=False, b_is_sparse=False
        )
        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias)
        if self.batchnorm:
            mean, variance = tf.nn.moments(x=x, axes=(0,), shift=None, keep_dims=False)
            x = tf.nn.batch_normalization(
                x=x, mean=mean, variance=variance, offset=self.offset, scale=self.scale,
                variance_epsilon=1e-6
            )
        if self.activation == 'none':
            pass
        elif self.activation == 'relu':
            x = tf.nn.relu(features=x)
        elif self.activation == 'softplus':
            x = tf.nn.softplus(features=x)
        elif self.activation == 'tanh':
            x = tf.tanh(x=x)
        else:
            raise NotImplementedError
        # dropout
        return x
