from typing import Optional, List, Dict
from collections import defaultdict

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

    def __init__(self, name: str, categories: Optional[List] = None):
        super().__init__(name=name)
        self.categories = categories
        self.idx_to_category = {0: np.nan}
        self.category_to_idx: Dict[str, int] = defaultdict(lambda: 0)

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", categories={self.categories})'

    def fit(self, x: pd.DataFrame) -> Transformer:

        if self.categories is None:
            categories = Nominal(self.name).extract(x).categories  # type: ignore
        else:
            categories = np.array(self.categories)

        try:
            categories.sort()
        except TypeError:
            pass

        # check for NaN and delete to put at front of category array
        try:
            categories = np.delete(categories, np.isnan(categories))
        except TypeError:
            pass
        categories = np.array([np.nan, *categories])

        for idx, cat in enumerate(categories[1:]):
            self.category_to_idx[cat] = idx + 1
            self.idx_to_category[idx + 1] = cat

        return super().fit(x)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        # convert NaN to str. Otherwise np.nan are used as dict keys, which can be dodgy
        x[self.name] = x[self.name].fillna('nan')
        x[self.name] = x[self.name].apply(lambda x: self.category_to_idx[x])
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = x[self.name].apply(lambda x: self.idx_to_category[x])
        return x

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'CategoricalTransformer':
        return cls(meta.name, meta.categories)  # type: ignore
