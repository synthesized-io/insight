import numpy as np
import tensorflow as tf

from ..module import tensorflow_name_scoped
from .functional import Functional


class ProbabilitiesFunctional(Functional):

    def __init__(self, probabilities, categories, value=None, values=None, name=None):
        if values is None:
            assert value is not None
            values = (value,)
        else:
            assert value is None

        super().__init__(values=values, name=name)

        self.probabilities = probabilities
        self.categories = categories

    def specification(self):
        spec = super().specification()
        spec.update(probabilities=self.probabilities, categories=self.categories)
        return spec

    @tensorflow_name_scoped
    def loss(self, samples):
        num_samples = tf.shape(input=samples)[0]
        samples = tf.concat(
            values=(tf.range(start=0, limit=len(self.probabilities)), samples), axis=0
        )
        _, _, counts = tf.unique_with_counts(x=samples)
        counts = counts - 1
        probs = tf.cast(x=counts, dtype=tf.float32) / tf.cast(x=num_samples, dtype=tf.float32)
        loss = tf.math.squared_difference(x=probs, y=self.probabilities)
        loss = tf.reduce_mean(input_tensor=loss, axis=0)
        return loss

    def check_distance(self, samples):
        return abs(
            np.asarray(np.sum(samples == category) / len(samples) for category in self.categories)
            - self.probabilities
        )
