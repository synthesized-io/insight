import tensorflow as tf
from synthesized.core import util
from synthesized.core.transformations import Transformation


class DenseTransformation(Transformation):

    def __init__(self, name, input_size, output_size, batchnorm=True, activation='relu'):
        super().__init__(name=name, input_size=input_size, output_size=output_size)
        self.batchnorm = batchnorm
        self.activation = activation

    def _initialize(self):
        self.weight = tf.get_variable(
            name='weight', shape=(self.input_size, self.output_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        self.bias = tf.get_variable(
            name='bias', shape=(self.output_size,), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        self.offset = tf.get_variable(
            name='offset', shape=(self.output_size,), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        self.scale = tf.get_variable(
            name='scale', shape=(self.output_size,), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )

    def transform(self, x):
        x = tf.matmul(a=x, b=self.weight, name=None)
        x = tf.nn.bias_add(value=x, bias=self.bias, name=None)
        if self.batchnorm:
            mean, variance = tf.nn.moments(x=x, axes=(0,), shift=None, name=None, keep_dims=False)
            x = tf.nn.batch_normalization(
                x=x, mean=mean, variance=variance, offset=self.offset, scale=self.scale,
                variance_epsilon=1e-6, name=None
            )
        if self.activation == 'none':
            pass
        elif self.activation == 'relu':
            x = tf.nn.relu(features=x, name=None)
        elif self.activation == 'tanh':
            x = tf.tanh(x=x, name=None)
        else:
            raise NotImplementedError
        # dropout
        return x
