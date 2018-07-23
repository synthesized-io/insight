from ..module import Module


class Value(Module):

    def __init__(self, name):
        super().__init__(name=name)

    def input_size(self, x=None):
        raise NotImplementedError

    def output_size(self):
        return self.input_size()

    def feature(self, x=None):
        raise NotImplementedError

    def tf_input_tensor(self, feed=None):
        raise NotImplementedError

    def tf_output_tensor(self, x):
        raise NotImplementedError

    def tf_loss(self, x, feed=None):
        raise NotImplementedError
