from collections import defaultdict
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from ..base import Transformer
from ...metadata import Nominal
from ...model import DiscreteModel


class CategoricalTransformer(Transformer):
    """
    Map nominal values onto integers.

    Attributes:
        name (str) : the data frame column to transform.
        categories (list, optional). list of unique categories, defaults to None
            If None, categories are extracted from the data.
    """

    def __init__(self, name: str, categories: Sequence = None, category_to_idx: Dict[Any, int] = None):
        super().__init__(name=name)
        self.categories = categories

        if categories is not None and category_to_idx is not None:
            raise ValueError("Only one of categories of categories_to_idx can be specified")

        if categories is not None:
            self._set_mapping_from_categories(categories)
        elif category_to_idx is not None:
            self._set_mapping(category_to_idx)
            if not isinstance(self.idx_to_category[0], float) or not np.isnan(self.idx_to_category[0]):
                raise ValueError("category to index mapping should only map nan values to 0")

    def _set_mapping_from_categories(self, categories: Sequence):
        """ Takes a list of categories and updates instance with a simple mapping from them """
        categories = np.array(categories)
        categories = np.delete(categories, pd.isna(categories).nonzero())
        self.categories = np.array([np.nan, *categories])  # type: ignore

        self.category_to_idx = defaultdict(int)
        self.idx_to_category = {0: np.nan}

        for idx, cat in enumerate(self.categories[1:]):  # type: ignore
            self.category_to_idx[cat] = idx + 1
            self.idx_to_category[idx + 1] = cat

    def _set_mapping(self, category_to_idx: Dict[Any, int]):
        """ Takes a mapping and updates the instance to use it """
        self.categories = list(category_to_idx.keys())
        self.category_to_idx = defaultdict(int, category_to_idx)
        self.idx_to_category = {0: np.nan}
        for category, idx in category_to_idx.items():
            self.idx_to_category[idx] = category

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", categories={self.categories})'

    def fit(self, df: pd.DataFrame) -> 'CategoricalTransformer':
        if self.categories is None:
            categories = df[self.name].unique()  # type: ignore
            self._set_mapping_from_categories(categories)

        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # convert NaN to str. Otherwise np.nan are used as dict keys, which can be dodgy
        df.loc[:, self.name] = df.loc[:, self.name].fillna('nan')
        df.loc[:, self.name] = df.loc[:, self.name].apply(lambda x: self.category_to_idx[x])
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.loc[:, self.name] = df.loc[:, self.name].apply(lambda x: self.idx_to_category[x])
        return df

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'CategoricalTransformer':
        return cls(meta.name, meta.categories)

    @classmethod
    def from_model(cls, model: DiscreteModel) -> 'CategoricalTransformer':
        return cls(model.name, model.categories)
