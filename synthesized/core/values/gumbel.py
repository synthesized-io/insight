from .continuous import ContinuousValue
from scipy.stats import norm
from scipy.stats import gumbel_r
import numpy as np


class GumbelDistrValue(ContinuousValue):

    def __init__(self, name, params = None):
        super().__init__(name=name, positive=False)
        self.params = params
        self.location = params[0]
        self.scale = params[1]

    def __str__(self):
        string = super().__str__()
        string += '-gumbel'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(location=self.location, scale=self.scale)
        return spec

    def extract(self, data):
        super().extract(data=data)
        if self.params is None:
            self.location = 1.
            self.scale = 1.

    def encode(self, data):
        return super().encode(data=data)

    def preprocess(self, data):
        data = super().preprocess(data=data)
        data[self.name] = data[self.name].apply(lambda x : norm.ppf(gumbel_r.cdf(x, self.location, self.scale)))
        data = data[data != float('inf')].dropna()
        return data

    def postprocess(self, data):
        data[self.name] = data[self.name].apply(lambda x : gumbel_r.ppf(norm.cdf(x), self.location, self.scale))
        return super().postprocess(data=data)
