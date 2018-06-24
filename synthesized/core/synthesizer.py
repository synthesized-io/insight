class Synthesizer(object):

    def __init__(self, dtypes):
        self.dtypes = dtypes

    def fit(self, data):
        raise NotImplementedError

    def synthesize(self, n):
        raise NotImplementedError
