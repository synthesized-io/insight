from synthesized.core import Module


class Encoding(Module):

    def __init__(self, name, encoding_size):
        super().__init__(name=name)
        self.encoding_size = encoding_size

    def encode(self, x, encoding_loss=False):
        raise NotImplementedError

    def sample(self, n):
        raise NotImplementedError
