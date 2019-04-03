from ..module import Module, tensorflow_name_scoped


class Transformation(Module):

    def __init__(self, name, input_size, output_size):
        super().__init__(name=name)

        self.input_size = input_size
        self.output_size = output_size

    def specification(self):
        spec = super().specification()
        spec.update(input_size=self.input_size, output_size=self.output_size)
        return spec

    def size(self):
        return self.output_size

    @tensorflow_name_scoped
    def transform(self, x, *args):
        raise NotImplementedError
