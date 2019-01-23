from .continuous import ContinuousValue
from scipy.stats import norm
from scipy.stats import gumbel_r


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
