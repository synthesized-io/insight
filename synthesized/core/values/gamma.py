from synthesized.core.module import tensorflow_name_scoped
from .continuous import ContinuousValue
from scipy.stats import norm
from scipy.stats import gamma
import tensorflow as tf
import tensorflow_probability as tfp

class GammaDistrValue(ContinuousValue):

    def __init__(self, name, integer=None, params=None):
        super().__init__(name=name, integer=integer)
        self.params = params
        self.shape = params[0]
        self.location = params[1]
        self.scale = params[2]

    def __str__(self):
        string = super().__str__()
        return string

    def specification(self):
        spec = super().specification()
        spec.update(shape=self.shape, location=self.location, scale=self.scale)
        return spec

    def extract(self, data):
        # super().extract(data=data)
        if self.params is None:
            self.shape = 1.
            self.scale = 1.
            self.location = 1.

    def preprocess(self, data):
        data = super().preprocess(data=data)
        data[self.name] = norm.ppf(gamma.cdf(data[self.name], self.shape, self.location, self.scale))
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return data

    def postprocess(self, data):
        data[self.name] = gamma.ppf(norm.cdf(data[self.name]),  self.shape, self.location, self.scale)
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return super().postprocess(data=data)

    @tensorflow_name_scoped
    def output_tensors(self, x):
        return super().output_tensors(x=x)

    @tensorflow_name_scoped
    def distribution_loss(self, samples):
        samples = tf.squeeze(input=samples, axis=1)
        tfd = tfp.distributions
        dist_normlal = tfd.Normal(loc=0., scale=1.)
        dist_gamma = tfd.Gamma(concentration=self.shape, rate=self.scale)
        samples = dist_gamma.cdf(value = samples)
        samples = dist_normlal.quantile(value = samples)
        samples = tf.boolean_mask(samples, tf.is_finite(samples))
        samples = tf.boolean_mask(samples, tf.math.logical_not(tf.is_nan(samples)))

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
        loss = mean_loss + variance_loss + jarque_bera_loss

        return loss
