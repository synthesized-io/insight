import tensorflow as tf

from .functional import Functional


class VarianceFunctional(Functional):

    def __init__(self, output, variance, name=None):
        super().__init__(outputs=(output,), name=name)

        self.variance = variance

    def specification(self):
        spec = super().specification()
        spec.update(variance=self.variance)
        return spec

    def tf_loss(self, samples):
        _, variance = tf.nn.moments(x=samples, axes=0)
        loss = tf.squared_difference(x=variance, y=self.variance)
        return loss
