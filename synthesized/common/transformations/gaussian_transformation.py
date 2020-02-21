import tensorflow as tf

from .transformation import Transformation
from .dense import DenseTransformation


class GaussianTransformation(Transformation):
    def __init__(
            self, input_size: int, output_size: int, name: str = 'gaussian-transformation'
    ):
        super(GaussianTransformation, self).__init__(name=name, input_size=input_size, output_size=output_size)

        self.mean = DenseTransformation(
            name='mean', input_size=input_size, output_size=output_size, batchnorm=False, activation='none'
        )
        self.stddev = DenseTransformation(
            name='stddev', input_size=input_size, output_size=output_size, batchnorm=False, activation='softplus'
        )

    def build(self, input_shape):
        self.mean.build(input_shape)
        self.stddev.build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        mu = self.mean(inputs, **kwargs)
        sigma = self.stddev(inputs, **kwargs)
        tf.summary.histogram(name='mean', data=mu)
        tf.summary.histogram(name='stddev', data=sigma)
        return mu, sigma
