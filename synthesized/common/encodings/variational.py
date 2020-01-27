import tensorflow as tf

from .encoding import Encoding
from ..transformations import DenseTransformation
from ..module import tensorflow_name_scoped


class VariationalEncoding(Encoding):

    def __init__(self, name, input_size, encoding_size, beta=None):
        super().__init__(name=name, input_size=input_size, encoding_size=encoding_size)
        self.beta = beta

        self.mean = DenseTransformation(
            name='mean', input_size=self.input_size,
            output_size=self.encoding_size, batchnorm=False, activation='none'
        )
        self.stddev = DenseTransformation(
            name='stddev', input_size=self.input_size,
            output_size=self.encoding_size, batchnorm=False, activation='softplus'
        )

    def build(self, input_shape):
        self.mean.build(input_shape)
        self.stddev.build(input_shape)
        self.built = True

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    def call(self, inputs, condition=()) -> tf.Tensor:
        mean = self.mean(inputs)
        stddev = self.stddev(inputs)
        x = tf.random.normal(
            shape=tf.shape(input=mean), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        x = mean + stddev * x

        encoding_loss = 0.5 * (tf.square(x=mean) + tf.square(x=stddev)) \
            - tf.math.log(x=tf.maximum(x=stddev, y=1e-6)) - 0.5
        encoding_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=encoding_loss, axis=1), axis=0)

        if self.beta is not None:
            encoding_loss *= self.beta

        self.add_loss(encoding_loss, inputs=inputs)

        return x

    @tensorflow_name_scoped
    def sample(self, n, condition=()):
        x = tf.random.normal(
            shape=(n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        return x
