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
        dist_gamma = tfd.Gamma(concentration=self.shape, rate=1.)
        samples = tf.where (samples < self.location, tf.add(samples, 2 * self.location), samples)
        samples = ( samples - self.location ) / self.scale
        samples = dist_gamma.cdf(value = samples) / self.scale
        samples = tf.boolean_mask(samples, tf.math.logical_not(tf.is_nan(samples)))
        samples = dist_normlal.quantile(value = samples)
        samples = tf.boolean_mask(samples, tf.is_finite(samples))
        samples = tf.boolean_mask(samples, tf.math.logical_not(tf.is_nan(samples)))
        return  super.distribution_loss(samples = samples)

