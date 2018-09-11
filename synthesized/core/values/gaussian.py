from .continuous import ContinuousValue


class GaussianValue(ContinuousValue):

    def __init__(self, name, mean=None, stddev=None):
        super().__init__(name=name, positive=False)
        self.mean = mean
        self.stddev = stddev

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

    def tf_input_tensor(self, feed=None):
        x = super().tf_input_tensor(feed=feed)
        x = (x - self.mean) / self.stddev
        return x

    def tf_output_tensors(self, x):
        x = super().tf_output_tensors(x=x)[self.name]
        x = x * self.stddev + self.mean
        return {self.name: x}
