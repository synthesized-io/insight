import tensorflow as tf
from synthesized.core.encodings import Encoding


class BasicEncoding(Encoding):

    def __init__(self, name, encoding_size, sampling='normal'):
        super().__init__(name=name, encoding_size=encoding_size)
        self.sampling = sampling

    def _initialize(self):
        self.weight = tf.get_variable(
            name='weight', shape=(self.encoding_size, self.encoding_size), dtype=tf.float32,
            initializer=None, regularizer=None, trainable=True, collections=None,
            caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
            custom_getter=None, constraint=None
        )
        self.bias = tf.get_variable(
            name='bias', shape=(self.encoding_size,), dtype=tf.float32,
            initializer=None, regularizer=None, trainable=True, collections=None,
            caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
            custom_getter=None, constraint=None
        )

    def encode(self, x, encoding_loss=False):
        encoded = tf.matmul(a=x, b=self.weight, name=None)
        encoded = tf.nn.bias_add(value=encoded, bias=self.bias, name=None)
        encoded = tf.tanh(x=encoded, name=None)
        return encoded

    def sample(self, n):
        if self.sampling == 'uniform':
            sampled = tf.random_uniform(
                shape=(n, self.encoding_size), minval=-1.0, maxval=1.0, dtype=tf.float32,
                seed=None, name=None
            )
        elif self.sampling == 'normal':
            sampled = tf.truncated_normal(
                shape=(n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32,
                seed=None, name=None
            )
        else:
            raise NotImplementedError
        return sampled
