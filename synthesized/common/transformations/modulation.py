from typing import Dict, Any

from .dense import DenseTransformation
from .transformation import Transformation
from ..module import tensorflow_name_scoped


class ModulationTransformation(Transformation):

    def __init__(self, name, input_size, condition_size):
        super().__init__(name=name, input_size=input_size, output_size=input_size)

        self.condition_size = condition_size

        self.offset = DenseTransformation(
            name='offset', input_size=self.condition_size,
            output_size=self.input_size, bias=False, batch_norm=False, activation='none'
        )
        self.scale = DenseTransformation(
            name='scale', input_size=self.condition_size,
            output_size=self.input_size, bias=False, batch_norm=False, activation='none'
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

    def get_variables(self) -> Dict[str, Any]:
        if not self.built:
            raise ValueError(self.name + " hasn't been built yet. ")

        variables = super().get_variables()
        variables.update(
            condition_size=self.condition_size,
            offset=self.offset.get_variables(),
            scale=self.scale.get_variables()
        )
        return variables

    def set_variables(self, variables: Dict[str, Any]):
        super().set_variables(variables)

        assert self.condition_size == variables['condition_size']

        if not self.built:
            self.build(self.input_size)

        self.offset.set_variables(variables['offset'])
        self.scale.set_variables(variables['scale'])
