import tensorflow as tf

from .encoding import Encoding
from ..transformations import DenseTransformation


class BasicEncoding(Encoding):

    def __init__(self, name, input_size, encoding_size, sampling='normal'):
        super().__init__(name=name, input_size=input_size, encoding_size=encoding_size)
        self.sampling = sampling

        self.embedding = self.add_module(
            module=DenseTransformation, name='embedding', input_size=self.input_size,
            output_size=self.encoding_size, batchnorm=True, activation='none'
        )

    def specification(self):
        spec = super().specification()
        spec.update(sampling=self.sampling)
        return spec

    def size(self):
        return self.encoding_size

    def tf_encode(self, x, encoding_loss=False):
        x = self.embedding.transform(x=x)
        if encoding_loss:
            return x, tf.constant(value=0.0, dtype=tf.float32)
        else:
            return x

    def tf_sample(self, n):
        if self.sampling == 'normal':
            x = tf.truncated_normal(
                shape=(n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
            )
        # elif self.sampling == 'uniform':
        #     x = tf.random_uniform(
        #         shape=(n, self.encoding_size), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=None
        #     )
        else:
            raise NotImplementedError
        return x
