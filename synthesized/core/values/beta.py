from .continuous import ContinuousValue
from scipy.stats import norm
from scipy.stats import beta


class BetaDistrValue(ContinuousValue):
    # TO DO
    def __init__(self, name, params=None):
        super().__init__(name=name)
        self.location = params[0]
        self.scale = params[1]

    def __str__(self):
        string = super().__str__()
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
        data[self.name] = norm.ppf(beta.cdf(data[self.name], self.location, self.scale))
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return data

    def postprocess(self, data):
        data[self.name] = beta.ppf(norm.cdf(data[self.name]), self.location, self.scale)
        data = data.dropna()
        data = data[data[self.name] != float('inf')]
        data = data[data[self.name] != float('-inf')]
        return super().postprocess(data=data)
