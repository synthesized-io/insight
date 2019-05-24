import tensorflow as tf

from .encoding import Encoding
from ..transformations import DenseTransformation
from ..module import tensorflow_name_scoped


class BasicEncoding(Encoding):

    def __init__(self, name, input_size, encoding_size, sampling='normal'):
        super().__init__(name=name, input_size=input_size, encoding_size=encoding_size)
        self.sampling = sampling

        self.embed = self.add_module(
            module=DenseTransformation, name='embedding', input_size=self.input_size,
            output_size=self.encoding_size, batchnorm=False, activation='none'
        )

    def specification(self):
        spec = super().specification()
        spec.update(sampling=self.sampling)
        return spec

    def size(self):
        return self.encoding_size

    @tensorflow_name_scoped
    def encode(self, x, encoding_loss=False):
        x = self.embed.transform(x=x)
        if encoding_loss:
            return x, tf.constant(value=0.0, dtype=tf.float32)
        else:
            return x

    @tensorflow_name_scoped
    def sample(self, n):
        if self.sampling == 'normal':
            x = tf.truncated_normal(shape=(n, self.encoding_size))
        elif self.sampling == 'uniform':
            x = tf.random_uniform(
                shape=(n, self.encoding_size), minval=-1.0, maxval=1.0, dtype=tf.float32
            )
        else:
            raise NotImplementedError
        return x
