from synthesized.core import Module


class Transformation(Module):

    def __init__(self, name, input_size, output_size):
        super().__init__(name=name)
        self.input_size = input_size
        self.output_size = output_size

    def transform(self, x):
        raise NotImplementedError
