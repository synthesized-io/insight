import tensorflow as tf
from ..module import tensorflow_name_scoped
from ..transformations import Transformation


class Encoding(Transformation):

    def __init__(self, name, input_size, encoding_size, condition_size=0):
        super().__init__(name=name, input_size=input_size, output_size=encoding_size, dtype=tf.float32)
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.condition_size = condition_size
        self.global_step = tf.summary.experimental.get_step()

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
    def sample(self, n, condition=(), identifier=None):
        raise NotImplementedError

    @tf.function
    @tensorflow_name_scoped
    def get_beta(self, beta_start=0., beta_end=1., t_start=0, t_end=500):
        assert beta_start <= beta_end and t_start <= t_end
        global_step = tf.cast(self.global_step, dtype=tf.float32)

        if global_step <= t_start:
            beta = beta_start
        elif global_step <= t_end:
            beta = (beta_end - beta_start) / (t_end - t_start) * (global_step - t_end) + beta_end
        else:
            beta = beta_end

        tf.summary.scalar(name='beta', data=beta)
        return beta
