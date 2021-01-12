from typing import Any, Dict

from .dense import DenseTransformation
from .transformation import Transformation


class MlpTransformation(Transformation):
    def __init__(self, name, input_size, layer_sizes, batch_norm=True, activation='relu'):
        super().__init__(name=name, input_size=input_size, output_size=layer_sizes[-1])

        self.layers = list()
        self.layer_sizes = layer_sizes
        self.batch_norm = batch_norm
        self.activation = activation
        previous_size = self.input_size
        for n, layer_size in enumerate(layer_sizes):
            layer = DenseTransformation(
                name=('layer' + str(n)), input_size=previous_size,
                output_size=layer_size, batch_norm=batch_norm, activation=activation
            )
            self.layers.append(layer)
            previous_size = layer_size

    def specification(self):
        spec = super().specification()
        spec.update(layers=[layer.specification() for layer in self.layers])
        return spec

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs=inputs, **kwargs)

        return inputs

    @property
    def regularization_losses(self):
        losses = [loss for layer in self.layers for loss in layer.regularization_losses]

        return losses

    def get_variables(self) -> Dict[str, Any]:
        if not self.built:
            raise ValueError(self.name + " hasn't been built yet. ")

        variables = super().get_variables()
        variables.update(
            layer_sizes=self.layer_sizes,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        for i, layer in enumerate(self.layers):
            variables['layer_{}'.format(i)] = layer.get_variables()

        return variables

    def set_variables(self, variables: Dict[str, Any]):
        super().set_variables(variables)

        assert self.layer_sizes == variables['layer_sizes']

        if not self.built:
            self.build(self.input_size)

        for i, layer in enumerate(self.layers):
            layer.set_variables(variables['layer_{}'.format(i)])
