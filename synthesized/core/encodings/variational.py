import tensorflow as tf

from .encoding import Encoding
from ..transformations import DenseTransformation


class VariationalEncoding(Encoding):

    def __init__(self, name, input_size, encoding_size, beta=5.0):
        super().__init__(name=name, input_size=input_size, encoding_size=encoding_size)
        self.beta = beta

        self.mean = self.add_module(
            module=DenseTransformation, name='mean', input_size=self.input_size,
            output_size=self.encoding_size, batchnorm=False, activation='none'
        )
        self.stddev = self.add_module(
            module=DenseTransformation, name='stddev', input_size=self.input_size,
            output_size=self.encoding_size, batchnorm=False, activation='softplus'  # 'none'
        )

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    def tf_encode(self, x, encoding_loss=False):
        mean = self.mean.transform(x=x)
        stddev = self.stddev.transform(x=x)
        x = tf.random_normal(
            shape=tf.shape(input=mean), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        x = mean + stddev * x
        if encoding_loss:
            encoding_loss = 0.5 * (tf.square(x=mean) + tf.square(x=stddev)) \
                - tf.log(x=tf.maximum(x=stddev, y=1e-6)) - 0.5
            encoding_loss = tf.reduce_sum(input_tensor=encoding_loss, axis=(0, 1), keepdims=False)
            encoding_loss *= self.beta
            tf.losses.add_loss(loss=encoding_loss, loss_collection=tf.GraphKeys.LOSSES)
        return x

    def tf_sample(self, n):
        x = tf.random_normal(
            shape=(n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        return x
