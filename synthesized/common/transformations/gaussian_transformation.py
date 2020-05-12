from typing import Tuple, Dict, Any

import tensorflow as tf

from .transformation import Transformation
from .dense import DenseTransformation
from ..module import tensorflow_name_scoped


class GaussianTransformation(Transformation):
    def __init__(
            self, input_size: int, output_size: int, name: str = 'gaussian-transformation'
    ):
        super(GaussianTransformation, self).__init__(name=name, input_size=input_size, output_size=output_size)

        self.mean = DenseTransformation(
            name='mean', input_size=input_size, output_size=output_size, batch_norm=False, activation='none'
        )
        self.stddev = DenseTransformation(
            name='stddev', input_size=input_size, output_size=output_size, batch_norm=False, activation='softplus'
        )

    @tensorflow_name_scoped
    def build(self, input_shape):
        self.mean.build(input_shape)
        self.stddev.build(input_shape)
        self.built = True

    @tf.function(input_signature=[tf.TensorSpec(name='inputs', shape=None, dtype=tf.float32)])
    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        mu = self.mean(inputs, **kwargs)
        sigma = self.stddev(inputs, **kwargs)
        tf.summary.histogram(name='mean', data=mu)
        tf.summary.histogram(name='stddev', data=sigma)
        return mu, sigma

    @property
    def regularization_losses(self):
        return [loss for layer in [self.mean, self.stddev] for loss in layer.regularization_losses]

    def get_variables(self) -> Dict[str, Any]:
        if not self.built:
            raise ValueError(self.name + " hasn't been built yet. ")

        variables = super().get_variables()
        variables.update(
            mean=self.mean.get_variables(),
            stddev=self.stddev.get_variables()
        )
        return variables

    def set_variables(self, variables: Dict[str, Any]):
        super().set_variables(variables)

        if not self.built:
            self.build(self.input_size)

        self.mean.set_variables(variables['mean'])
        self.stddev.set_variables(variables['stddev'])
