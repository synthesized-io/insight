import tensorflow as tf
from ..module import tensorflow_name_scoped


class Encoding(tf.keras.layers.Layer):

    def __init__(self, name, input_size, encoding_size, condition_size=0):
        super().__init__(name=name, dtype=tf.float32)
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.condition_size = condition_size

    def specification(self):
        spec = dict()
        spec.update(
            input_size=self.input_size, encoding_size=self.encoding_size,
            condition_size=self.condition_size
        )
        return spec

    def size(self):
        raise NotImplementedError

    @tensorflow_name_scoped
    def call(self, x, condition=()):
        raise NotImplementedError

    @tensorflow_name_scoped
    def sample(self, n, condition=()):
        raise NotImplementedError
