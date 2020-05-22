from typing import Dict, Any

import tensorflow as tf

from ..module import tensorflow_name_scoped
from ..transformations import Transformation
from ..util import check_format_version


class Encoding(Transformation):
    format_version = '0.0'

    def __init__(self, input_size, encoding_size, condition_size=0, name='encoding'):
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
    def increase_beta_multiplier(self, t_start: tf.Tensor = 0, t_end: tf.Tensor = 500):
        """Multiply beta by this to obtain a linear increase from zero to beta.

        Example:
            >>> beta = beta * self.increase_beta_multiplier()

        """
        global_step = tf.cast(self.global_step, dtype=tf.float32)

        if global_step <= t_start:
            beta = 0.
        elif global_step <= t_end:
            beta = (global_step - t_start) / (t_end - t_start)
        else:
            beta = 1.

        tf.summary.scalar(name='beta', data=beta)
        return beta

    @staticmethod
    def diagonal_normal_kl_divergence(mu_1: tf.Tensor, stddev_1: tf.Tensor,
                                      mu_2: tf.Tensor = None, stddev_2: tf.Tensor = None):
        if mu_2 is None:
            mu_2 = tf.zeros(shape=mu_1.shape, dtype=tf.float32)

        if stddev_2 is None:
            stddev_2 = tf.ones(shape=stddev_1.shape, dtype=tf.float32)

        cov_1 = tf.maximum(x=tf.square(stddev_1), y=1e-6)
        cov_2 = tf.maximum(x=tf.square(stddev_2), y=1e-6)

        return 0.5 * tf.reduce_mean(
            tf.reduce_sum(
                tf.math.log(cov_2 / cov_1) + (tf.square(mu_1 - mu_2) + cov_1 - cov_2) / cov_2,
                axis=-1
            )
        )

    def get_variables(self) -> Dict[str, Any]:
        variables = super().get_variables()
        variables.update(
            format_version=self.format_version,
            encoding_size=self.encoding_size,
            condition_size=self.condition_size,
            global_step=self.global_step.numpy(),
        )
        return variables

    def set_variables(self, variables: Dict[str, Any]):
        check_format_version(self.format_version, variables['format_version'])

        super().set_variables(variables)
        assert self.encoding_size == variables['encoding_size']
        assert self.condition_size == variables['condition_size']

        self.global_step.assign(variables['global_step'])
