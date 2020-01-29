from .transformation import Transformation
from .residual import ResidualTransformation
from ..module import tensorflow_name_scoped


class ResnetTransformation(Transformation):

    def __init__(
        self, name, input_size, layer_sizes, depths=2, batchnorm=True, activation='relu'
    ):
        super().__init__(name=name, input_size=input_size, output_size=layer_sizes[-1])

        self.layers = list()
        previous_size = self.input_size
        depths = 2 if depths is None else depths

        for n, layer_size in enumerate(layer_sizes):
            if isinstance(depths, int):
                layer = ResidualTransformation(
                    name=('ResidualLayer_' + str(n)), input_size=previous_size, output_size=layer_size,
                    depth=depths, batchnorm=batchnorm, activation=activation
                )
            else:
                layer = ResidualTransformation(
                    name=('ResidualLayer_' + str(n)), input_size=previous_size, output_size=layer_size,
                    depth=depths[n], batchnorm=batchnorm, activation=activation
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

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    @property
    def regularization_losses(self):
        return [loss for layer in self.layers for loss in layer.regularization_losses]
