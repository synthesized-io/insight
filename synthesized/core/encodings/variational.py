import tensorflow as tf
from synthesized.core import util
from synthesized.core.encodings import Encoding


class VariationalEncoding(Encoding):

    def __init__(self, name, encoding_size, beta=5.0):
        super().__init__(name=name, encoding_size=encoding_size)
        self.beta = beta

    def _initialize(self):
        self.mean_weight = tf.get_variable(
            name='mean_weight', shape=(self.encoding_size, self.encoding_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        self.mean_bias = tf.get_variable(
            name='mean_bias', shape=(self.encoding_size,), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        self.stddev_weight = tf.get_variable(
            name='stddev_weight', shape=(self.encoding_size, self.encoding_size), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )
        self.stddev_bias = tf.get_variable(
            name='stddev_bias', shape=(self.encoding_size,), dtype=tf.float32,
            initializer=util.initializers['normal'], regularizer=util.regularizers['l2'],
            trainable=True, collections=None, caching_device=None, partitioner=None,
            validate_shape=True, use_resource=None, custom_getter=None
        )

    def encode(self, x, encoding_loss=False):
        assert x.shape[1].value == self.encoding_size
        mean = tf.matmul(a=x, b=self.mean_weight, name=None)
        mean = tf.nn.bias_add(value=mean, bias=self.mean_bias, name=None)
        stddev = tf.matmul(a=x, b=self.stddev_weight, name=None)
        stddev = tf.nn.bias_add(value=stddev, bias=self.stddev_bias, name=None)
        stddev = tf.nn.softplus(features=stddev, name=None)
        encoded = tf.random_normal(
            shape=tf.shape(input=x), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
            name=None
        )
        encoded = mean + stddev * encoded
        if encoding_loss:
            encoding_loss = 0.5 * (tf.square(x=mean, name=None) + tf.square(x=stddev, name=None)) \
                - tf.log(x=tf.maximum(x=stddev, y=1e-6, name=None), name=None) - 0.5
            encoding_loss = tf.reduce_sum(
                input_tensor=encoding_loss, axis=1, keepdims=False, name=None
            )
            encoding_loss = tf.reduce_mean(
                input_tensor=encoding_loss, axis=0, keepdims=False, name=None
            )
            encoding_loss *= self.beta
            tf.losses.add_loss(loss=encoding_loss, loss_collection=tf.GraphKeys.LOSSES)
        return encoded

    def sample(self, n):
        sampled = tf.random_normal(
            shape=(n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
            name=None
        )
        return sampled
