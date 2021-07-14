import re

import numpy as np
import pandas as pd

from synthesized.metadata import ValueMeta
from synthesized.transformer.base import Transformer
from synthesized.transformer.exceptions import NonInvertibleTransformError


class PartialTransformer(Transformer):
    """
    Transforms by masking out the first 75% (or N%) of each sample for the given column 'x'.

    Examples:
        "4905 9328 9320 4630" -> "xxxx xxxx xxxx 4630"

    Attributes:
        name (str) : the data frame column to transform.
        masking_proportion (float) : proportion of data to be masked, default is 0.75
    """

    def __init__(self, name: str, masking_proportion: float = 0.75):
        super().__init__(name=name)
        self.masking_proportion = masking_proportion

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}")'

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the given dataframe to the transformer

        Args:
            df: Dataset to fit

        Returns:
            self
        """
        return super().fit(df)

    def _mask_key(self, k):
        to_replace = int(np.ceil(len(k) * self.masking_proportion))
        regex = "^.{" + str(to_replace) + "}"
        return re.sub(regex, "x" * to_replace, k)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transforms the given dataframe using fitted transformer

        Args:
            df: Dataset to transform

        Returns:
            Transformed dataset
        """
        df.loc[:, self.name] = df.loc[:, self.name].astype(str).apply(self._mask_key)
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NonInvertibleTransformError

    @classmethod
    def from_meta(cls, meta: ValueMeta) -> 'PartialTransformer':
        return cls(meta.name)
