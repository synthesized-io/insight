from .continuous import ContinuousValue
from scipy.stats import norm, weibull_min


class WeibullDistrValue(ContinuousValue):

    def __init__(self, name, params=None):
        super().__init__(name=name)
        self.params = params
        self.shape = params[0]
        self.location = params[1]
        self.scale = params[2]

    def __str__(self):
        string = super().__str__()
        return string

    def specification(self):
        spec = super().specification()
        spec.update(scale=self.scale, location = self.location, shape=self.shape)
        return spec

    def extract(self, data):
        if self.params is None:
            self.scale = 1.
            self.shape = 1.
            self.location = 1.

    def preprocess(self, data):
        data = super().preprocess(data=data)
        data[self.name] = norm.ppf(weibull_min.cdf(data[self.name], self.shape, self.location, self.scale))
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return data

    def postprocess(self, data):
        data[self.name] = weibull_min.ppf(norm.cdf(data[self.name]), self.shape, self.location, self.scale)
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return super().postprocess(data=data)