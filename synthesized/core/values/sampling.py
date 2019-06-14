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
        self.categories = data[self.name].value_counts(normalize=True, sort=True, dropna=False)
        self.categories **= self.smoothing
        self.categories /= self.categories.sum()

    def postprocess(self, data):
        data.loc[:, self.name] = np.random.choice(
            a=self.categories.index, size=len(data), p=self.categories.values
        )
        return data
