from ..module import Module


class Classifier(Module):

    def __init__(self, name, summarizer=False):
        super().__init__(name=name, summarizer=summarizer)

    def learn(self, iterations, data=None, filenames=None, verbose=False):
        raise NotImplementedError

    def classify(self, data):
        raise NotImplementedError
