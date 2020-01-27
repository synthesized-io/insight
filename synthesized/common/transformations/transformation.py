from typing import Optional
import tensorflow as tf


class Transformation(tf.keras.layers.Layer):

    def __init__(self, name, input_size, output_size, dtype=tf.float32):
        super(Transformation, self).__init__(name=name, dtype=dtype)

        self.input_size = input_size
        self.output_size = output_size
        self._output: Optional[tf.Tensor] = None

    def specification(self):
        spec = dict(name=self.name)
        spec.update(input_size=self.input_size, output_size=self.output_size)
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
