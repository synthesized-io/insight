from .continuous import ContinuousValue


class GaussianValue(ContinuousValue):

    def __init__(self, name, mean=None, stddev=None):
        super().__init__(name=name, positive=False)
        self.mean = mean
        self.stddev = stddev

    def __str__(self):
        string = super().__str__()
        string += '-gaussian'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(mean=self.mean, stddev=self.stddev)
        return spec

    def extract(self, data):
        super().extract(data=data)
        if self.mean is None:
            self.mean = data[self.name].mean()
        if self.stddev is None:
            self.stddev = data[self.name].std()

    def preprocess(self, data):
        data = (data - self.mean) / self.stddev
        return super().preprocess(data=data)

    def postprocess(self, data):
        data = super().postprocess(data=data)
        data = data * self.stddev + self.mean
        return data
