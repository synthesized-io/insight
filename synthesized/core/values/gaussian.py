import tensorflow as tf

from .continuous import ContinuousValue


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

    def tf_input_tensor(self, feed=None):
        x = super().tf_input_tensor(feed=feed)
        x = (x - self.mean) / self.stddev
        return x

    def tf_output_tensors(self, x):
        x = x * self.stddev + self.mean
        return super().tf_output_tensors(x=x)

    def tf_distribution_loss(self, samples):
        samples = tf.squeeze(input=samples, axis=1)
        mean, variance = tf.nn.moments(x=samples, axes=0)
        loss = tf.squared_difference(x=mean, y=0.0) + tf.squared_difference(x=variance, y=1.0)
        return loss
