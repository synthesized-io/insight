from typing import Dict, List

import pandas as pd

from ..base import Transformer
from ...model.models import GenderModel


class GenderTransformer(Transformer):
    """
    Transform a continuous distribution using the quantile transform.

    Attributes:
        name (str) : the data frame column to transform.
    """

    def __init__(self, name: str, collections: Dict[str, List[str]]):
        super().__init__(name=name)
        self.collections = collections
        self._fitted = True

    def __repr__(self):
        return (f'{self.__class__.__name__}(name="{self.name}", collections="{self.collections}")')

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        value_map = {
            v: key
            for key, collection in self.collections.items()
            for v in collection
        }
        df[self.name] = df[self.name].astype(dtype=str).map(value_map)
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

    @classmethod
    def from_meta(cls, meta: GenderModel) -> 'GenderTransformer':
        return cls(meta.name, collections=meta.collections)

    @property
    def in_columns(self) -> List[str]:
        in_columns = [self.name]
        return in_columns
