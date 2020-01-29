from typing import Optional, List, Union
import tensorflow as tf


class Transformation(tf.keras.layers.Layer):

    def __init__(self, name, input_size, output_size, dtype=tf.float32):
        super(Transformation, self).__init__(name=name, dtype=dtype)

        self.input_size = input_size
        self.output_size = output_size
        self._output: Optional[tf.Tensor] = None
        self._regularization_losses: List[tf.Tensor] = list()

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

    @property
    def regularization_losses(self):
        return self._regularization_losses

    def add_l2_loss(self, variable: Union[tf.Variable, List[tf.Variable]]) -> Union[tf.Tensor, List[tf.Tensor]]:
        l2 = tf.keras.regularizers.l2(1.0)
        if type(variable) is list:
            l2_losses = [l2(v) for v in variable]
            self._regularization_losses.extend(l2_losses)
            return l2_losses
        else:
            l2 = tf.keras.regularizers.l2(1.0)
            l2_loss = l2(variable)
            self._regularization_losses.append(l2_loss)
            return l2_loss
