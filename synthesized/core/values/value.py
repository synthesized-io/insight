from ..module import Module


class Value(Module):

    def __init__(self, name):
        super().__init__(name=name)

    def input_size(self):
        raise NotImplementedError

    def output_size(self):
        return self.input_size()

    def labels(self):
        return
        yield

    def placeholders(self):
        return
        yield

    def preprocess(self, data):
        return data

    def postprocess(self, data):
        return data

    def feature(self, x=None):
        return None

    def tf_input_tensor(self, feed=None):
        return None

    def tf_output_tensors(self, x):
        return dict()

    def tf_loss(self, x, feed=None):
        return None
