from .continuous import ContinuousValue
from scipy.stats import norm
from scipy.stats import gamma
import numpy as np


class GammaDistrValue(ContinuousValue):

    def __init__(self, name, params = None):
        super().__init__(name=name, positive=False)
        self.params = params
        self.shape = params[0]
        self.location = params[1]
        self.scale = params[2]

    def __str__(self):
        string = super().__str__()
        string += '-gamma'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(shape=self.shape, location=self.location, scale=self.scale)
        return spec

    def extract(self, data):
        super().extract(data=data)
        if self.params is None:
            self.shape = 1.
            self.scale = 1.
            self.location = 1.

    def preprocess(self, data):
        data = super().preprocess(data=data)
        data[self.name] = data[self.name].apply(lambda x : norm.ppf(gamma.cdf(x, self.shape, self.location, self.scale)))
        data = data[data != float('inf')].dropna()
        return data

    def postprocess(self, data):
        data[self.name] = data[self.name].apply(lambda x : gamma.ppf(norm.cdf(x),  self.shape, self.location, self.scale))
        return super().postprocess(data=data)