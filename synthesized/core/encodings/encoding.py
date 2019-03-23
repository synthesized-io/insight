from ..module import Module


class Encoding(Module):

    def __init__(self, name, input_size, encoding_size, condition_size=0):
        super().__init__(name=name)
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.condition_size = condition_size

    def specification(self):
        spec = super().specification()
        spec.update(
            input_size=self.input_size, encoding_size=self.encoding_size,
            condition_size=self.condition_size
        )
        return spec

    def size(self):
        raise NotImplementedError

    def tf_encode(self, x, condition=(), encoding_loss=False):
        raise NotImplementedError

    def tf_sample(self, n, condition=()):
        raise NotImplementedError
