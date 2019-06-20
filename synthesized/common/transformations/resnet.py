from .residual import ResidualTransformation
from .transformation import Transformation
from ..module import tensorflow_name_scoped


class ResnetTransformation(Transformation):

    def __init__(self, name, input_size, layer_sizes, weight_decay, depths=2):
        super().__init__(name=name, input_size=input_size, output_size=layer_sizes[-1])

        self.layer_sizes = layer_sizes
        self.weight_decay = weight_decay
        self.depths = depths

        self.layers = list()
        previous_size = self.input_size
        for n, layer_size in enumerate(self.layer_sizes):
            if isinstance(self.depths, int):
                layer = self.add_module(
                    module=ResidualTransformation, name=('layer' + str(n)),
                    input_size=previous_size, output_size=layer_size, depth=self.depths,
                    weight_decay=weight_decay
                )
            else:
                layer = self.add_module(
                    module=ResidualTransformation, name=('layer' + str(n)),
                    input_size=previous_size, output_size=layer_size, depth=self.depths[n],
                    weight_decay=weight_decay
                )
            self.layers.append(layer)
            previous_size = layer_size

    def specification(self):
        spec = super().specification()
        spec.update(
            layer_sizes=list(self.layer_sizes), weight_decay=self.weight_decay, depths=self.depths
        )
        return spec

    @tensorflow_name_scoped
    def transform(self, x):
        for layer in self.layers:
            x = layer.transform(x=x)

        return x
