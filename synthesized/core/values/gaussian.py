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

        mean, variance = tf.nn.moments(x=samples, axes=0)
        mean_loss = tf.squared_difference(x=mean, y=0.0)
        variance_loss = tf.squared_difference(x=variance, y=1.0)

        mean = tf.stop_gradient(input=tf.reduce_mean(input_tensor=samples, axis=0))
        difference = samples - mean
        squared_difference = tf.square(x=difference)
        variance = tf.reduce_mean(input_tensor=squared_difference, axis=0)
        third_moment = tf.reduce_mean(input_tensor=(squared_difference * difference), axis=0)
        fourth_moment = tf.reduce_mean(input_tensor=tf.square(x=squared_difference), axis=0)
        skewness = third_moment / tf.pow(x=variance, y=1.5)
        kurtosis = fourth_moment / tf.square(x=variance)
        num_samples = tf.cast(x=tf.shape(input=samples)[0], dtype=tf.float32)
        # jarque_bera = num_samples / 6.0 * (tf.square(x=skewness) + \
        #     0.25 * tf.square(x=(kurtosis - 3.0)))
        jarque_bera = tf.square(x=skewness) + tf.square(x=(kurtosis - 3.0))
        jarque_bera_loss = tf.squared_difference(x=jarque_bera, y=0.0)

        return mean_loss + variance_loss + jarque_bera_loss


