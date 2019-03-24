from .dense import DenseTransformation
from .transformation import Transformation
from ..module import tensorflow_name_scoped


class MlpTransformation(Transformation):

    def __init__(self, name, input_size, layer_sizes, weight_decay):
        super().__init__(name=name, input_size=input_size, output_size=layer_sizes[-1])

        self.layer_sizes = layer_sizes
        self.weight_decay = weight_decay

        self.layers = list()
        previous_size = self.input_size
        for n, layer_size in enumerate(self.layer_sizes):
            layer = self.add_module(
                module=DenseTransformation, name=('layer' + str(n)), input_size=previous_size,
                output_size=layer_size, weight_decay=weight_decay
            )
            self.layers.append(layer)
            previous_size = layer_size

    def specification(self):
        spec = super().specification()
        spec.update(layer_sizes=list(self.layer_sizes), weight_decay=self.weight_decay)
        return spec

    @tensorflow_name_scoped
    def transform(self, x):
        for layer in self.layers:
            x = layer.transform(x=x)

        return x
