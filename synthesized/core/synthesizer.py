from sklearn.base import BaseEstimator, TransformerMixin
from synthesized.core import Module


class Synthesizer(Module, TransformerMixin):

    def __init__(self, name, submodules=()):
        super().__init__(name=name, submodules=submodules, master=True)

    def learn(self, data, verbose=False):
        raise NotImplementedError

    def synthesize(self, n):
        raise NotImplementedError

    def transform(self, X, **transform_params):
        raise NotImplementedError

    def fit(self, X, y=None, **fit_params):
        assert y is None and not fit_params
        self.learn(data=X)
        return self
