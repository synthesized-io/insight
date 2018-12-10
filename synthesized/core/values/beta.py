from .continuous import ContinuousValue
from scipy.stats import norm
from scipy.stats import beta
import numpy as np


class BetaDistrValue(ContinuousValue):
    # TO DO
    def __init__(self, name, params = None):
        super().__init__(name=name, positive=True)
    #NOT SURE ABOUT positive=True, depends on how it's processed further down the line
        self.location = params[0]
        self.scale = params[1]

    def __str__(self):
        string = super().__str__()
        string += '-gumbel'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(exponent=self.location, scale=self.scale)
        return spec

    def extract(self, data):
        super().extract(data=data)
        if self.params is None:
            self.location = 1.
            self.scale = 1.

    def preprocess(self, data):
        data = super().preprocess(data=data)
        data[self.name] = data[self.name].apply(lambda x : norm.ppf(beta.cdf(x, self.location, self.scale)))
        data = data[data != float('inf')].dropna()
        return data

    def postprocess(self, data):
        data[self.name] = data[self.name].apply(lambda x : beta.ppf(norm.cdf(x), self.location, self.scale))
        return super().postprocess(data=data)
