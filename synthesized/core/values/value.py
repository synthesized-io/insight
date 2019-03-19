from ..module import Module, tensorflow_name_scoped


class Value(Module):

    def __init__(self, name):
        super().__init__(name=name)

    def __str__(self):
        return self.__class__.__name__[:-5].lower()

    def input_size(self):
        return 0

    def output_size(self):
        return self.input_size()

    def input_labels(self):
        if self.input_size() > 0:
            yield self.name

    def output_labels(self):
        yield from self.input_labels()

    def placeholders(self):
        return
        yield

    def extract(self, data):
        pass

    def encode(self, data):
        return data

    def preprocess(self, data):
        return self.encode(data)

    def postprocess(self, data):
        return data

    def features(self, x=None):
        return dict()

    @tensorflow_name_scoped
    def input_tensor(self, feed=None):
        return None

    @tensorflow_name_scoped
    def output_tensors(self, x):
        return dict()

    @tensorflow_name_scoped
    def loss(self, x, feed=None):
        return None

    @tensorflow_name_scoped
    def distribution_loss(self, samples):
        return None
