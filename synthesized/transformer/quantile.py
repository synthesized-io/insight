from typing import Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

from .base import Transformer
from ..metadata_new import Float, Integer


class QuantileTransformer(Transformer):
    """
    Transform a continuous distribution using the quantile transform.

    Attributes:
        name (str) : the data frame column to transform.
        n_quantiles (int, optional). Number of quantiles to compute, defaults to 1000.
        output_distribution (str). Marginal distribution for the transformed data.
            Either 'uniform' or 'normal', defaults to 'normal'.
    """

    def __init__(self, name: str, n_quantiles: int = 1000, output_distribution: str = 'normal', noise: float = 1e-7):
        super().__init__(name=name)
        self._transformer = _QuantileTransformer(n_quantiles, output_distribution)
        self.noise = noise

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", n_quantiles={self._transformer.n_quantiles}, output_distribution="{self._transformer.output_distribution}", noise={self.noise})'

    def fit(self, df: pd.DataFrame) -> 'QuantileTransformer':
        if len(df) < self._transformer.n_quantiles:
            self._transformer = self._transformer.set_params(n_quantiles=len(df))

        if self.noise:
            df[self.name] += np.random.normal(loc=0, scale=self.noise, size=(len(df)))

        self._transformer.fit(df[[self.name]])
        return super().fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.noise:
            df[self.name] += np.random.normal(loc=0, scale=self.noise, size=(len(df)))

        positive = (df[self.name] > 0.0).all()
        nonnegative = (df[self.name] >= 0.0).all()

        if nonnegative and not positive:
            df[self.name] = np.maximum(df[self.name], 0.001)

        df[self.name] = self._transformer.transform(df[[self.name]]).astype(np.float32)

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = self._transformer.inverse_transform(df[[self.name]])
        return df

    @classmethod
    def from_meta(cls, meta: Union[Float, Integer]) -> 'QuantileTransformer':
        return cls(meta.name)
