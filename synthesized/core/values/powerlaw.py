from .continuous import ContinuousValue


class PowerlawValue(ContinuousValue):

    def __init__(self, name, exponent=None, scale=None, offset=None):
        super().__init__(name=name, positive=True)
        self.exponent = exponent
        self.scale = scale
        self.offset = offset

    def __str__(self):
        string = super().__str__()
        string += '-powerlaw'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(exponent=self.exponent, scale=self.scale, offset=self.offset)
        return spec

    def extract(self, data):
        super().extract(data=data)
        if self.exponent is None:
            self.exponent = 0.5
        if self.scale is None:
            self.scale = 1.0
        if self.offset is None:
            self.offset = 0.0

    def preprocess(self, data):
        # bug?
        data = ((data - self.offset) / self.scale) ** self.exponent
        return super().preprocess(data=data)

    def postprocess(self, data):
        # bug?
        data = super().postprocess(data=data)
        data = (data ** (1.0 / self.exponent)) * self.scale + self.offset
        return data
