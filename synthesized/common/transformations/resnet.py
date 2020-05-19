from typing import Dict, Any

import tensorflow as tf

from .residual import ResidualTransformation
from .transformation import Transformation
from ..module import tensorflow_name_scoped
from ..util import check_params_version


class ResnetTransformation(Transformation):

    def __init__(self, name, input_size, layer_sizes, depths=2, batch_norm=True, activation='relu'):
        super().__init__(name=name, input_size=input_size, output_size=layer_sizes[-1])
        self.params_version = '0.0'

        self.layer_sizes = layer_sizes
        self.depths = depths

        self.layers = list()
        previous_size = self.input_size
        depths = 2 if depths is None else depths

        for n, layer_size in enumerate(layer_sizes):
            if isinstance(depths, int):
                layer = ResidualTransformation(
                    name=('ResidualLayer_' + str(n)), input_size=previous_size, output_size=layer_size,
                    depth=depths, batch_norm=batch_norm, activation=activation
                )
            else:
                layer = ResidualTransformation(
                    name=('ResidualLayer_' + str(n)), input_size=previous_size, output_size=layer_size,
                    depth=depths[n], batch_norm=batch_norm, activation=activation
                )
            self.layers.append(layer)
            previous_size = layer_size

    def specification(self):
        spec = super().specification()
        spec.update(layers=[layer.specification() for layer in self.layers])
        return spec

    @tensorflow_name_scoped
    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.built = True

    @tf.function(input_signature=[tf.TensorSpec(name='inputs', shape=None, dtype=tf.float32)])
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    @property
    def regularization_losses(self):
        return [loss for layer in self.layers for loss in layer.regularization_losses]

    def get_variables(self) -> Dict[str, Any]:
        if not self.built:
            raise ValueError(self.name + " hasn't been built yet. ")

        variables = super().get_variables()
        variables.update(
            params_version=self.params_version,
            layer_sizes=self.layer_sizes,
            depths=self.depths
        )
        for i, layer in enumerate(self.layers):
            variables['residual_layer_{}'.format(i)] = layer.get_variables()

        return variables

    def set_variables(self, variables: Dict[str, Any]):
        check_params_version(self.params_version, variables['params_version'])

        super().set_variables(variables)

        assert self.layer_sizes == variables['layer_sizes']
        assert self.depths == variables['depths']

        if not self.built:
            self.build(self.input_size)

        for i, layer in enumerate(self.layers):
            layer.set_variables(variables['residual_layer_{}'.format(i)])
