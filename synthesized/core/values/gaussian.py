import tensorflow as tf

from .continuous import ContinuousValue
from ..module import tensorflow_name_scoped


class GaussianValue(ContinuousValue):

    def __init__(self, name, mean=None, stddev=None):
        super().__init__(name=name, positive=False)
        self.mean = mean
        self.stddev = stddev

    def __str__(self):
        string = super().__str__()
        string += '-gaussian'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(mean=self.mean, stddev=self.stddev)
        return spec

    def extract(self, data):
        super().extract(data=data)
        if self.mean is None:
            self.mean = data[self.name].mean()
        if self.stddev is None:
            self.stddev = data[self.name].std()

    @tensorflow_name_scoped
    def input_tensor(self, feed=None):
        x = super().input_tensor(feed=feed)
        x = (x - self.mean) / self.stddev
        return x

    @tensorflow_name_scoped
    def output_tensors(self, x):
        x = x * self.stddev + self.mean
        return super().output_tensors(x=x)

    @tensorflow_name_scoped
    def distribution_loss(self, samples):
        samples = tf.squeeze(input=samples, axis=1)
        return super.distribution_loss(samples = samples)
