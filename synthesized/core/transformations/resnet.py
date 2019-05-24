from .transformation import Transformation
from ..module import tensorflow_name_scoped


class ResnetTransformation(Transformation):

    def __init__(
        self, name, input_size, layer_sizes, depths=2, layer_type='dense', batchnorm=True,
        activation='relu', weight_decay=0.0
    ):
        super().__init__(name=name, input_size=input_size, output_size=layer_sizes[-1])

        from . import transformation_modules
        self.layers = list()
        previous_size = self.input_size
        for n, layer_size in enumerate(layer_sizes):
            if isinstance(depths, int):
                layer = self.add_module(
                    module='residual', modules=transformation_modules, name=('layer' + str(n)),
                    input_size=previous_size, output_size=layer_size, depth=depths,
                    layer_type=layer_type, batchnorm=batchnorm, activation=activation,
                    weight_decay=weight_decay
                )
            else:
                layer = self.add_module(
                    module='residual', modules=transformation_modules, name=('layer' + str(n)),
                    input_size=previous_size, output_size=layer_size, depth=depths[n],
                    layer_type=layer_type, batchnorm=batchnorm, activation=activation,
                    weight_decay=weight_decay
                )
            self.layers.append(layer)
            previous_size = layer_size

    def specification(self):
        spec = super().specification()
        spec.update(layers=[layer.specification() for layer in self.layers])
        return spec

    @tensorflow_name_scoped
    def transform(self, x):
        for layer in self.layers:
            x = layer.transform(x=x)

        return x
