from typing import Optional, List, Union, Dict, Any

import tensorflow as tf

from ..util import check_format_version


class Transformation(tf.keras.layers.Layer):
    format_version = '0.0'

    def __init__(self, name, input_size, output_size, dtype=tf.float32):
        super(Transformation, self).__init__(name=name, dtype=dtype)

        self.input_size = input_size
        self.output_size = output_size
        self._output: Optional[tf.Tensor] = None
        self._regularization_losses: List[tf.Tensor] = list()

    def specification(self):
        spec = dict(
            name=self.name,
            input_size=self.input_size,
            output_size=self.output_size
        )
        return spec

    def size(self):
        return self.output_size

    def compute_output_shape(self, input_shape):
        return [None, self.output_size]

    # @tensorflow_name_scoped
    # def transform(self, x, *args):
    #     raise NotImplementedError

    @property
    def output(self):
        return self._output

    @property
    def regularization_losses(self):
        return self._regularization_losses

    def add_regularization_weights(self, variable: Union[tf.Variable, List[tf.Variable]]) -> \
            Union[tf.Variable, List[tf.Variable]]:
        if type(variable) is list:
            self._regularization_losses.extend(variable)
        else:
            self._regularization_losses.append(variable)
        return variable

    def get_variables(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            format_version=self.format_version,
            input_size=self.input_size,
            output_size=self.output_size
        )

    def set_variables(self, variables: Dict[str, Any]):
        check_format_version(self.format_version, variables['format_version'])

        assert self.name == variables['name']
        assert self.input_size == variables['input_size']
        assert self.output_size == variables['output_size']
