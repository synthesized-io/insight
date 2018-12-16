from .continuous import ContinuousValue
from scipy.stats import norm
from scipy.stats import gilbrat


class GilbratDistrValue(ContinuousValue):

    def __init__(self, name, params=None):
        super().__init__(name=name)
        self.params = params
        self.shape = params[0]
        self.location = params[1]

    def __str__(self):
        string = super().__str__()
        return string

    def specification(self):
        spec = super().specification()
        spec.update(shape=self.shape, location=self.location)
        return spec

    def extract(self, data):
        # super().extract(data=data)
        if self.params is None:
            self.shape = 1.
            self.scale = 1.

    def preprocess(self, data):
        data = super().preprocess(data=data)
        data[self.name] = norm.ppf(gilbrat.cdf(data[self.name], self.shape, self.location))
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return data

    def postprocess(self, data):
        data[self.name] = gilbrat.ppf(norm.cdf(data[self.name]),  self.shape, self.location)
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return super().postprocess(data=data)
