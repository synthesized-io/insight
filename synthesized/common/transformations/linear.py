from .dense import DenseTransformation


class LinearTransformation(DenseTransformation):

    def __init__(self, name, input_size, output_size, bias=True, weight_decay=0.0):
        super(LinearTransformation, self).__init__(
            name=name, input_size=input_size, output_size=output_size, bias=bias, batchnorm=False,
            activation='none', weight_decay=weight_decay
        )
