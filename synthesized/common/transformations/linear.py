from .dense import DenseTransformation


class LinearTransformation(DenseTransformation):

    def __init__(self, name, input_size, output_size, bias=True):
        super(LinearTransformation, self).__init__(
            name=name, input_size=input_size, output_size=output_size, bias=bias, batch_norm=False,
            activation='none'
        )
