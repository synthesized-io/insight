import tensorflow as tf
from synthesized.core.transformations import DenseTransformation
from synthesized.core.encodings import Encoding


class BasicEncoding(Encoding):

    def __init__(self, name, input_size, encoding_size, sampling='uniform'):
        super().__init__(name=name, input_size=input_size, encoding_size=encoding_size)
        self.sampling = sampling

        self.embedding = self.add_module(
            module=DenseTransformation, name='embedding', input_size=self.input_size,
            output_size=self.encoding_size, batchnorm=False, activation='none'
        )

    def specification(self):
        spec = super().specification()
        spec.update(sampling=self.sampling)
        return spec

    def size(self):
        return self.encoding_size

    def tf_encode(self, x, encoding_loss=False):
        x = self.embedding.transform(x=x)
        x = tf.tanh(x=x, name=None)
        return x

    def tf_sample(self, n):
        if self.sampling == 'uniform':
            x = tf.random_uniform(
                shape=(n, self.encoding_size), minval=-1.0, maxval=1.0, dtype=tf.float32,
                seed=None, name=None
            )
        elif self.sampling == 'normal':
            x = tf.truncated_normal(
                shape=(n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32,
                seed=None, name=None
            )
        else:
            raise NotImplementedError
        return x
