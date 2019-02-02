import tensorflow as tf

from .functional import Functional


class MeanFunctional(Functional):

    def __init__(self, output, mean, name=None):
        super().__init__(outputs=(output,), name=name)

        self.mean = mean

    def specification(self):
        spec = super().specification()
        spec.update(mean=self.mean)
        return spec

    def tf_loss(self, samples):
        mean = tf.reduce_mean(input_tensor=samples, axis=0)
        loss = tf.squared_difference(x=mean, y=self.mean)
        return loss
