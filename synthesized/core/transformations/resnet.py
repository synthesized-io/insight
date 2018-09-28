from .residual import ResidualTransformation
from .transformation import Transformation


class ResnetTransformation(Transformation):

    def __init__(self, name, input_size, layer_sizes=(), depths=None):
        super().__init__(name=name, input_size=input_size, output_size=layer_sizes[-1])
        self.layer_sizes = layer_sizes
        self.depths = depths

        self.layers = list()
        previous_size = self.input_size
        for n, layer_size in enumerate(self.layer_sizes):
            if self.depths is None:
                layer = self.add_module(
                    module=ResidualTransformation, name=('layer' + str(n)),
                    input_size=previous_size, output_size=layer_size
                )
            else:
                layer = self.add_module(
                    module=ResidualTransformation, name=('layer' + str(n)),
                    input_size=previous_size, output_size=layer_size, depth=self.depths[n]
                )
            self.layers.append(layer)
            previous_size = layer_size

    def specification(self):
        spec = super().specification()
        spec.update(layer_sizes=list(self.layer_sizes), depths=self.depths)
        return spec

    def tf_transform(self, x):
        for layer in self.layers:
            x = layer.transform(x=x)
        return x
