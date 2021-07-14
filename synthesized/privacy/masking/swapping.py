from typing import Optional

import numpy as np
import pandas as pd

from synthesized.metadata import Nominal
from synthesized.transformer.base import Transformer
from synthesized.transformer.exceptions import NonInvertibleTransformError


class SwappingTransformer(Transformer):
    """
    Transforms by shuffling the categories around for a given column.

    Examples:
        'a' -> 'b' ('a' and 'b' are both existing values.)

    Attributes:
        name (str) : the data frame column to transform.
        uniform (bool) : if True, then distribute all the categories uniformly,
                         if False, then maintain their existing proportion
    """

    def __init__(self, name: str, uniform: bool = False):
        super().__init__(name=name)
        self.uniform = uniform
        self.categories: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the given dataframe to the transformer

        Args:
            df: Dataset to fit

        Returns:
            self
        """
        self.categories = df.loc[:, self.name].value_counts(normalize=True, sort=True, dropna=False)
        if self.uniform:
            p = 1 / len(self.categories)
            self.categories = self.categories.apply(lambda x: p)

        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transforms the given dataframe using fitted transformer

        Args:
            df: Dataset to transform

        Returns:
            Transformed dataset
        """
        assert self.categories is not None
        df.loc[:, self.name] = np.random.choice(
            a=self.categories.index, size=len(df), p=self.categories.values
        )
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NonInvertibleTransformError

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'SwappingTransformer':
        return cls(meta.name)
