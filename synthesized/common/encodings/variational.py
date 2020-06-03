from typing import Dict, Any

import tensorflow as tf
import tensorflow_probability as tfp

from .encoding import Encoding
from ..module import tensorflow_name_scoped
from ..transformations import GaussianTransformation


class VariationalEncoding(Encoding):
    def __init__(self, input_size, encoding_size, beta=1.0, name='variational_encoding'):
        super().__init__(name=name, input_size=input_size, encoding_size=encoding_size)
        self.beta = beta

        self.gaussian = GaussianTransformation(input_size=input_size, output_size=encoding_size)

    @tensorflow_name_scoped
    def build(self, input_shape):
        self.gaussian.build(input_shape)
        self.built = True

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    def call(self, inputs, condition=()) -> tf.Tensor:
        mean, stddev = self.gaussian(inputs)

        x = tf.random.normal(
            shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        x = mean + stddev * x

        kl_loss = self.diagonal_normal_kl_divergence(mean, stddev)
        # kl_loss = tf.multiply(self.beta, kl_loss)
        kl_loss = self.beta * kl_loss

        # self.add_loss(lambda: kl_loss, inputs=inputs)
        self.add_loss(kl_loss, inputs=inputs)
        tf.summary.histogram(name='mean', data=mean),
        tf.summary.histogram(name='stddev', data=stddev),
        tf.summary.histogram(name='posterior_distribution', data=x),
        tf.summary.image(
            name='latent_space_correlation',
            data=tf.abs(tf.reshape(tfp.stats.correlation(x), shape=(1, self.encoding_size, self.encoding_size, 1)))
        )

        return x

    @tensorflow_name_scoped
    def sample(self, n) -> tf.Tensor:
        z = tf.random.normal(
            shape=(n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        return z

    @property
    def regularization_losses(self):
        return [loss for layer in [self.gaussian] for loss in layer.regularization_losses]

    def get_variables(self) -> Dict[str, Any]:
        if not self.built:
            raise ValueError(self.name + " hasn't been built yet. ")

        variables = super().get_variables()
        variables.update(
            beta=self.beta,
            gaussian=self.gaussian.get_variables()
        )
        return variables

    def set_variables(self, variables: Dict[str, Any]):
        super().set_variables(variables)

        if not self.built:
            self.build(self.input_size)

        self.beta = variables['beta']
        self.gaussian.set_variables(variables['gaussian'])
