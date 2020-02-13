import pandas as pd
import tensorflow as tf

from .state_space import StateSpaceModel
from ..values import ValueFactory, ValueOps
from ..transformations import Transformation, MlpTransformation, DenseTransformation


class FeedForwardStateSpaceModel(StateSpaceModel):
    """A Deep State Space model using only feed forward networks"""
    def __init__(self, df: pd.DataFrame, capacity: int, latent_size: int, name: str = 'ff-state-space-model'):
        super(FeedForwardStateSpaceModel, self).__init__(self, capacity=capacity, latent_size=latent_size, name=name)

        self.value_factory = ValueFactory(df=df, capacity=capacity)
        self.value_ops = ValueOps(
            values=self.value_factory.get_values(), conditions=self.value_factory.get_conditions()
        )

        self.emission_network = GaussianEncoder(
            input_size=latent_size, output_size=self.value_ops.output_size, capacity=capacity,
            num_layers=1
        )
        self.transition_network = GaussianEncoder(
            input_size=latent_size+self.value_ops.output_size, output_size=latent_size, capacity=capacity,
            num_layers=4
        )
        self.inference_network = GaussianEncoder(
            input_size=latent_size + 2*self.value_ops.output_size, output_size=latent_size, capacity=capacity,
            num_layers=4
        )
        self.initial_network = GaussianEncoder(
            input_size=self.value_ops.input_size, output_size=latent_size, capacity=capacity,
            num_layers=1
        )

    def build(self, input_shape):
        self.emission_network.build(self.latent_size)
        self.transition_network.build(self.latent_size + self.value_ops.output_size)
        self.inference_network.build(self.latent_size + 2*self.value_ops.output_size)
        self.initial_network.build(self.value_ops.input_size)
        self.built = True

    def regularization_losses(self):
        pass


class GaussianEncoder(Transformation):
    def __init__(
            self, input_size: int, output_size: int, capacity: int, num_layers: int, name: str = 'gaussian-encoder'
    ):
        super(GaussianEncoder, self).__init__(name=name)

        self.network = MlpTransformation(
            name='mlp-encoder', input_size=input_size, layer_sizes=[capacity for _ in range(num_layers)],
            batchnorm=False
        )
        self.mean = DenseTransformation(
            name='mean', input_size=capacity,
            output_size=output_size, batchnorm=False, activation='none'
        )
        self.stddev = DenseTransformation(
            name='stddev', input_size=capacity,
            output_size=output_size, batchnorm=False, activation='softplus'
        )

    def build(self, input_shape):
        self.network.build(input_shape)
        self.mean.build(self.network.output_size)
        self.stddev.build(self.network.output_size)
        self.built = True

    def call(self, inputs, **kwargs):
        z = self.network(inputs)
        mu, sigma = self.mean(z), self.stddev(z)

        return mu, sigma
