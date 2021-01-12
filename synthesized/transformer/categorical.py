from collections import defaultdict
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .base import Transformer
from ..metadata_new import Nominal


class CategoricalTransformer(Transformer):
    """
    Map nominal values onto integers.

    Attributes:
        name (str) : the data frame column to transform.
        categories (list, optional). list of unique categories, defaults to None
            If None, categories are extracted from the data.
    """

    def __init__(self, name: str, categories: Optional[Sequence[Any]] = None):
        super().__init__(name=name)
        self.categories = categories
        self.idx_to_category = {0: np.nan}
        self.category_to_idx: Dict[str, int] = defaultdict(lambda: 0)

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", categories={self.categories})'

    def fit(self, df: pd.DataFrame) -> 'CategoricalTransformer':

        if self.categories is None:
            categories = df[self.name].unique()  # type: ignore
        else:
            categories = np.array(self.categories)

        try:
            categories = np.delete(categories, np.isnan(categories))
        except TypeError:
            pass
        categories = np.array([np.nan, *categories])  # type: ignore

        for idx, cat in enumerate(categories[1:]):  # type: ignore
            self.category_to_idx[cat] = idx + 1
            self.idx_to_category[idx + 1] = cat

        return super().fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert NaN to str. Otherwise np.nan are used as dict keys, which can be dodgy
        df[self.name] = df[self.name].fillna('nan')
        df[self.name] = df[self.name].apply(lambda x: self.category_to_idx[x])
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = df[self.name].apply(lambda x: self.idx_to_category[x])
        return df

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'CategoricalTransformer':
        return cls(meta.name, meta.categories)
