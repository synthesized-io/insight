from synthesized.core import Module


class Value(Module):

    def __init__(self, name):
        super().__init__(name=name)

    def size(self):
        raise NotImplementedError

    def input_tensor(self):
        raise NotImplementedError

    def output_tensor(self, x):
        raise NotImplementedError

    def loss(self, x):
        raise NotImplementedError
