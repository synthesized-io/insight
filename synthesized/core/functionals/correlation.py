import tensorflow as tf

from .functional import Functional


class CorrelationFunctional(Functional):

    def __init__(self, output1, output2, correlation, name=None):
        assert output1 != output2

        super().__init__(outputs=(output1, output2), name=name)

        self.correlation = correlation

    def specification(self):
        spec = super().specification()
        spec.update(correlation=self.correlation)
        return spec

    def tf_loss(self, samples1, samples2):
        mean1, variance1 = tf.nn.moments(x=samples1, axes=0)
        mean2, variance2 = tf.nn.moments(x=samples2, axes=0)
        mean1 = tf.stop_gradient(input=mean1)
        mean2 = tf.stop_gradient(input=mean2)
        covariance = tf.reduce_mean(input_tensor=((samples1 - mean1) * (samples2 - mean2)), axis=0)
        correlation = covariance / tf.sqrt(x=variance1) / tf.sqrt(x=variance2)
        loss = tf.squared_difference(x=correlation, y=self.correlation)
        return loss
