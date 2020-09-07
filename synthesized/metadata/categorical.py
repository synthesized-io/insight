from collections.abc import MutableSequence
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .value_meta import ValueMeta


class CategoricalMeta(ValueMeta):

    def __init__(
        self, name: str,
        # Optional
        similarity_based: bool = False, pandas_category: bool = False, produce_nans: bool = False,
        # Scenario
        categories: List = None, probabilities=None, true_categorical: bool = True
    ):
        super().__init__(name=name)
        self._categories: Optional[MutableSequence] = None
        self.category2idx: Optional[Dict] = None
        self.idx2category: Optional[Dict] = None
        self.nans_valid: bool = False
        self.num_categories: Optional[int] = None

        self.given_categories = categories
        self.probabilities = probabilities

        self.similarity_based = similarity_based
        self.pandas_category = pandas_category
        self.produce_nans = produce_nans
        self.true_categorical = true_categorical
        self.is_string = False

    @property
    def categories(self):
        if self.nans_valid:
            return self._categories

        if self.is_string:
            return [c for c in self._categories if c != 'nan']
        else:
            return [c for c in self._categories if not np.isnan(c)]

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            num_categories=self.num_categories, similarity_based=self.similarity_based,
        )
        return spec

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        if df.loc[:, self.name].dtype.kind == 'O':
            self.is_string = True
            unique_values = df.loc[:, self.name].astype(object).fillna('nan').apply(str).unique().tolist()
        else:
            unique_values = df.loc[:, self.name].fillna(np.nan).unique().tolist()
        self._set_categories(unique_values)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_string:
            df.loc[:, self.name] = df.loc[:, self.name].astype(object).fillna('nan').apply(str)

        assert isinstance(self._categories, list) and isinstance(self.num_categories, int)
        df.loc[:, self.name] = df.loc[:, self.name].map(self.category2idx)

        if df.loc[:, self.name].dtype != 'int64':
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='int64')
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(self._categories, list)
        df.loc[:, self.name] = df.loc[:, self.name].map(self.idx2category)
        if self.is_string:
            df.loc[df[self.name] == 'nan', self.name] = np.nan

        if self.pandas_category:
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='category')
        self.set_dtypes(df)
        return df

    def _set_categories(self, categories: List):

        if self.given_categories is None:
            # Put any nan at the position zero of the list
            for n, x in enumerate(categories):
                if self.is_string and x == 'nan':
                    self.nans_valid = True
                    categories.remove(x)
                    break
                elif isinstance(x, float) and np.isnan(x):
                    self.nans_valid = True
                    categories.remove(x)
                    break

            try:
                categories = list(np.sort(categories))
            except TypeError:
                pass  # Don't sort the categories if it's not possible.

        else:
            for x in categories:
                if x not in self.given_categories:
                    self.nans_valid = True
                    break

            categories = self.given_categories.copy()

        nan = 'nan' if self.is_string else np.nan
        categories.insert(0, nan)

        # If categories are not set
        if self._categories is None:
            self._categories = categories
            self.num_categories = len(categories)
            self.idx2category = {i: self._categories[i] for i in range(len(self._categories))}
            self.category2idx = Categories({self._categories[i]: i for i in range(len(self._categories))})

        # If categories have been set and are different to the given
        elif isinstance(self._categories, list):
            for category in categories[int(self.nans_valid):]:
                if category not in self._categories[int(self.nans_valid):]:
                    raise NotImplementedError


class Categories(dict):

    def __missing__(self, key):
        return 0
