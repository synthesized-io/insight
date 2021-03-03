from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

from ..base import Transformer
from ...config import QuantileTransformerConfig
from ...metadata_new import Affine


class QuantileTransformer(Transformer):
    """
    Transform a continuous distribution using the quantile transform.

    Attributes:
        name (str) : the data frame column to transform.
        n_quantiles (int, optional). Number of quantiles to compute, defaults to 1000.
        output_distribution (str). Marginal distribution for the transformed data.
            Either 'uniform' or 'normal', defaults to 'normal'.
    """

    def __init__(self, name: str, config: Optional[QuantileTransformerConfig] = None):
        super().__init__(name=name)
        config = QuantileTransformerConfig() if config is None else config
        self._transformer = _QuantileTransformer(
            n_quantiles=config.n_quantiles, output_distribution=config.distribution
        )
        self.noise = config.noise

    def __repr__(self):
        return (f'{self.__class__.__name__}(name="{self.name}", n_quantiles={self._transformer.n_quantiles}, '
                f'output_distribution="{self._transformer.output_distribution}", noise={self.noise})')

    def fit(self, df: pd.DataFrame) -> 'QuantileTransformer':
        if len(df) < self._transformer.n_quantiles:
            self._transformer = self._transformer.set_params(n_quantiles=len(df))

        self._positive = (df[self.name] > 0.0).all()
        self._nonnegative = (df[self.name] >= 0.0).all()

        column = df.loc[:, self.name].astype('float32')

        if self.noise:
            column += np.random.normal(loc=0, scale=self.noise, size=(len(df)))

        self._transformer.fit(column.to_numpy().reshape(-1, 1))
        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.noise:
            df.loc[:, self.name] += np.random.normal(loc=0, scale=self.noise, size=(len(df)))

        if self._nonnegative and not self._positive:
            df.loc[df[self.name] < 0.001, self.name] = 0.001

        df[self.name] = self._transformer.transform(df[[self.name]]).astype('float32')

        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.loc[:, self.name] = self._transformer.inverse_transform(df[[self.name]]).astype('float32')
        if self._nonnegative:
            df.loc[(df.loc[:, self.name] < 0.001), self.name] = 0
        return df

    @classmethod
    def from_meta(cls, meta: Affine, config: Optional[QuantileTransformerConfig] = None) -> 'QuantileTransformer':
        return cls(meta.name, config=config)
