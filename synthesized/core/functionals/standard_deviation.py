import numpy as np
import tensorflow as tf

from .functional import Functional
from ..module import tensorflow_name_scoped


class StandardDeviationFunctional(Functional):

    def __init__(self, stddev, value=None, values=None, name=None):
        if values is None:
            assert value is not None
            values = (value,)
        else:
            assert value is None and len(values) == 1

        super().__init__(values=values, name=name)

        self.stddev = stddev

    def specification(self):
        spec = super().specification()
        spec.update(stddev=self.stddev)
        return spec

    @tensorflow_name_scoped
    def loss(self, samples):
        _, variance = tf.nn.moments(x=samples, axes=0)
        stddev = tf.sqrt(x=variance)
        loss = tf.squared_difference(x=stddev, y=self.stddev)
        return loss

    def check_distance(self, samples):
        return abs(np.std(samples) - self.stddev)
