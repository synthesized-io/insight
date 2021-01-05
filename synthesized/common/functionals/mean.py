import numpy as np
import tensorflow as tf

from ..module import tensorflow_name_scoped
from .functional import Functional


class MeanFunctional(Functional):

    def __init__(self, mean, value=None, values=None, name=None):
        if values is None:
            assert value is not None
            values = (value,)
        else:
            assert value is None and len(values) == 1

        super().__init__(values=values, name=name)

        self.mean = mean

    def specification(self):
        spec = super().specification()
        spec.update(mean=self.mean)
        return spec

    @tensorflow_name_scoped
    def loss(self, samples):
        mean = tf.reduce_mean(input_tensor=samples, axis=0)
        loss = tf.math.squared_difference(x=mean, y=self.mean)
        return loss

    def check_distance(self, samples):
        return abs(np.mean(samples) - self.mean)
