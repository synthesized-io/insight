from .continuous import ContinuousValue
from scipy.stats import norm
from scipy.stats import gumbel_r
from ..module import tensorflow_name_scoped
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class GumbelDistrValue(ContinuousValue):

    def __init__(self, name, integer=None, params=None):
        super().__init__(name=name, integer=integer)
        self.params = params
        self.location = params[0]
        self.scale = params[1]

    def __str__(self):
        string = super().__str__()
        return string

    def specification(self):
        spec = super().specification()
        spec.update(location=self.location, scale=self.scale)
        return spec

    def extract(self, data):
        if self.params is None:
            self.location = 1.
            self.scale = 1.

    def encode(self, data):
        return super().encode(data=data)

    def preprocess(self, data):
        data = super().preprocess(data=data)
        data[self.name] = norm.ppf(gumbel_r.cdf(data[self.name], self.location, self.scale))
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return data

    def postprocess(self, data):
        data[self.name] = gumbel_r.ppf(norm.cdf(data[self.name]), self.location, self.scale)
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return super().postprocess(data=data)

    @tensorflow_name_scoped
    def distribution_loss(self, samples):
        samples = tf.squeeze(input=samples, axis=1)
        tfd = tfp.distributions
        dist_normlal = tfd.Normal(loc=0., scale=1.)
        dist_gumbel = tfd.Gumbel(loc=self.location, scale = self.scale)
        samples = tf.where (samples < self.location, tf.add(samples, 2 * self.location), samples)
        samples = dist_gumbel.cdf(value = samples)
        samples = tf.boolean_mask(samples, tf.math.logical_not(tf.is_nan(samples)))
        samples = dist_normlal.quantile(value = samples)
        samples = tf.boolean_mask(samples, tf.is_finite(samples))
        samples = tf.boolean_mask(samples, tf.math.logical_not(tf.is_nan(samples)))
        return  super.distribution_loss(samples = samples)