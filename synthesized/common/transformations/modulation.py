from .dense import DenseTransformation
from .transformation import Transformation
from ..module import tensorflow_name_scoped


class ModulationTransformation(Transformation):

    def __init__(self, name, input_size, condition_size):
        super().__init__(name=name, input_size=input_size, output_size=input_size)

        self.condition_size = condition_size

        self.offset = DenseTransformation(
            name='offset', input_size=self.condition_size,
            output_size=self.input_size, bias=False, batchnorm=False, activation='none'
        )
        self.scale = DenseTransformation(
            name='scale', input_size=self.condition_size,
            output_size=self.input_size, bias=False, batchnorm=False, activation='none'
        )

    def specification(self):
        spec = super().specification()
        spec.update(condition_size=self.condition_size)
        return spec

    def build(self, input_shape):
        self.scale.build(input_shape)
        self.offset.build(input_shape)
        self.built = True

    @tensorflow_name_scoped
    def call(self, inputs, condition):
        offset = self.offset(inputs=condition)
        scale = self.scale(inputs=condition)
        inputs = inputs * scale + offset
        return inputs
