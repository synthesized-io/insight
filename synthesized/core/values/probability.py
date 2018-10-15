import numpy as np

from .categorical import CategoricalValue


class ProbabilityValue(CategoricalValue):

    def __init__(
        self, name, embedding_size, granularity=0.01, similarity_based=False, temperature=1.0,
        smoothing=0.1, moving_average=True, similarity_regularization=0.01,
        entropy_regularization=0.01
    ):
        if granularity > 0.5 or granularity < 0.001:
            raise NotImplementedError

        categories = [
            min(n * granularity, 1.0)
            for n in range(1.0 // granularity + (1.0 % granularity > 0.0))
        ]
        super().__init__(
            name=name, embedding_size=embedding_size, categories=categories, pandas_category=False,
            similarity_based=similarity_based, temperature=temperature, smoothing=smoothing,
            moving_average=moving_average, similarity_regularization=similarity_regularization,
            entropy_regularization=entropy_regularization
        )

        self.granularity = granularity

    def __str__(self):
        string = super().__str__()
        string += '-probability'
        return string

    def extract(self, data):
        if self.categories is None:
            raise NotImplementedError
        elif (data[self.name] < 0.0).any() or (data[self.name] > 1.0).any():
            raise NotImplementedError

    def preprocess(self, data):
        decimal_granularity = 1.0
        for n in range(4):
            if decimal_granularity == self.granularity:
                data[self.name] = data[self.name].round(decimals=n)
                break
            decimal_granularity *= 0.1
        else:
            data[self.name] = ((data[self.name] / self.granularity).round() * self.granularity) \
                .clip(upper=1.0)
        super().preprocess(data=data)

    def postprocess(self, data):
        super().postprocess(data=data)
        noise = np.random.random(len(data)) * self.granularity - 0.5 * self.granularity
        data[self.name] = (data[self.name] + noise).clip(lower=0.0, upper=1.0)
        return data
