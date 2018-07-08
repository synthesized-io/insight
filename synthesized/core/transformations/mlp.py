from synthesized.core.transformations import Transformation, DenseTransformation


class MlpTransformation(Transformation):

    def __init__(self, name, input_size, output_size, layer_sizes=()):
        super().__init__(name=name, input_size=input_size, output_size=output_size)
        self.layers = list()
        previous_size = self.input_size
        for n, layer_size in enumerate(layer_sizes):
            layer = self.add_module(
                module=DenseTransformation, name=('layer' + str(n)), input_size=previous_size,
                output_size=layer_size
            )
            self.layers.append(layer)
            previous_size = layer_size
        layer = self.add_module(
            module=DenseTransformation, name=('layer' + str(len(layer_sizes))),
            input_size=previous_size, output_size=self.output_size, batchnorm=False,
            activation='none'
        )
        self.layers.append(layer)

    def transform(self, x):
        for layer in self.layers:
            x = layer.transform(x=x)
        return x
