import numpy as np
import tensorflow as tf

from ..module import tensorflow_name_scoped
from .functional import Functional


class CorrelationFunctional(Functional):

    def __init__(self, correlation, value1=None, value2=None, values=None, name=None):
        if values is None:
            assert value1 is not None and value2 is not None and value1 != value2
            values = (value1, value2)
        else:
            assert value1 is None and value2 is None and len(values) == 2

        super().__init__(values=values, name=name)

        self.correlation = correlation

    def specification(self):
        spec = super().specification()
        spec.update(correlation=self.correlation)
        return spec

    @tensorflow_name_scoped
    def loss(self, samples1, samples2):
        mean1, variance1 = tf.nn.moments(x=samples1, axes=0)
        mean2, variance2 = tf.nn.moments(x=samples2, axes=0)
        mean1 = tf.stop_gradient(input=mean1)
        mean2 = tf.stop_gradient(input=mean2)
        covariance = tf.reduce_mean(input_tensor=((samples1 - mean1) * (samples2 - mean2)), axis=0)
        correlation = covariance / tf.sqrt(x=variance1) / tf.sqrt(x=variance2)
        loss = tf.math.squared_difference(x=correlation, y=self.correlation)
        return loss

    def check_distance(self, samples1, samples2):
        return abs(
            np.cov([samples1, samples2])[0, 1] / np.std(samples1) / np.std(samples2)
            - self.correlation
        )
