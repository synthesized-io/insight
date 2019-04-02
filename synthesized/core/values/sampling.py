import numpy as np

from .value import Value


class SamplingValue(Value):

    def __init__(self, name, uniform=False, smoothing=None):
        super().__init__(name=name)

        if uniform:
            if smoothing is not None and smoothing != 0.0:
                raise NotImplementedError
            self.smoothing = 0.0
        elif smoothing is None:
            self.smoothing = 1.0
        else:
            self.smoothing = smoothing

    def specification(self):
        spec = super().specification()
        spec.update(smoothing=self.smoothing)
        return spec

    def extract(self, data):
        if self.smoothing == 0.0:
            self.categories = data[self.name].unique()

        else:
            self.categories = data[self.name].value_counts(normalize=True, sort=True, dropna=False)
            if self.smoothing != 1.0:
                self.categories **= self.smoothing
                self.categories /= self.categories.sum()
            self.categories = self.categories.cumsum()
            if (self.categories.tail(n=1).astype(dtype='float32') != 1.0).bool():
                raise NotImplementedError

            def sample(p):
                for category, cumulative_probability in self.categories.iteritems():
                    if p <= cumulative_probability:
                        return category
                raise NotImplementedError

            self.sample = np.vectorize(pyfunc=sample)

    def postprocess(self, data):
        if self.smoothing == 0.0:
            data[self.name] = np.random.choice(a=self.categories, size=len(data))
        else:
            data[self.name] = self.sample(p=np.random.uniform(size=len(data)))
        return data
