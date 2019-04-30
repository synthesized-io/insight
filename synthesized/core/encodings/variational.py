import tensorflow as tf

from .encoding import Encoding
from ..transformations import transformation_modules
from ..module import tensorflow_name_scoped


class VariationalEncoding(Encoding):

    def __init__(self, name, input_size, encoding_size, condition_size=0, beta=None):
        super().__init__(
            name=name, input_size=input_size, encoding_size=encoding_size,
            condition_size=condition_size
        )
        self.beta = beta

        self.mean = self.add_module(
            module='dense', modules=transformation_modules, name='mean',
            input_size=self.input_size, output_size=self.encoding_size, batchnorm=False,
            activation='none'
        )
        self.stddev = self.add_module(
            module='dense', modules=transformation_modules, name='stddev',
            input_size=self.input_size, output_size=self.encoding_size, batchnorm=False,
            activation='softplus'
        )
        self.decoder = self.add_module(
            module='dense', modules=transformation_modules, name='initial-input',
            input_size=(self.encoding_size + self.condition_size), output_size=self.encoding_size,
            batchnorm=False, activation='none'
        )

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    @tensorflow_name_scoped
    def encode(self, x, condition=(), encoding_plus_loss=False):
        mean = self.mean.transform(x=x)
        stddev = self.stddev.transform(x=x)
        encoding = tf.random_normal(
            shape=tf.shape(input=mean), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        encoding = mean + stddev * encoding

        x = tf.concat(values=((encoding,) + tuple(condition)), axis=1)
        x = self.decoder.transform(x=x)

        if encoding_plus_loss:
            encoding_loss = 0.5 * (tf.square(x=mean) + tf.square(x=stddev)) \
                - tf.log(x=tf.maximum(x=stddev, y=1e-6)) - 0.5
            encoding_loss = tf.reduce_sum(input_tensor=encoding_loss, axis=(0, 1), keepdims=False)
            if self.beta is not None:
                encoding_loss *= self.beta
            return x, encoding, encoding_loss
        else:
            return x

    @tensorflow_name_scoped
    def sample(self, n, condition=()):
        encoding = tf.random_normal(
            shape=(n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )

        condition = tuple(tf.tile(input=c, multiples=(n, 1)) for c in condition)
        x = tf.concat(values=((encoding,) + tuple(condition)), axis=1)
        x = self.decoder.transform(x=x)

        return x
