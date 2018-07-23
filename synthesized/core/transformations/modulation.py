from ..transformations import Transformation, DenseTransformation


class ModulationTransformation(Transformation):

    def __init__(self, name, input_size, condition_size):
        super().__init__(name=name, input_size=input_size, output_size=input_size)
        self.condition_size = condition_size

        self.offset = self.add_module(
            module=DenseTransformation, name='offset', input_size=self.condition_size,
            output_size=self.input_size, bias=False, batchnorm=False, activation='none'
        )
        self.scale = self.add_module(
            module=DenseTransformation, name='scale', input_size=self.condition_size,
            output_size=self.input_size, bias=False, batchnorm=False, activation='none'
        )

    def specification(self):
        spec = super().specification()
        spec.update(condition_size=self.condition_size)
        return spec

    def tf_transform(self, x, condition):
        offset = self.offset.transform(x=condition)
        scale = self.scale.transform(x=condition)
        x = x * scale + offset
        return x
