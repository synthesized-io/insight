from typing import Dict, Optional, Sequence, cast

import pandas as pd

from ..base import Ordinal, ValueMeta
from ..exceptions import ExtractionError


class OrderedString(Ordinal[str]):
    dtype: str = 'U'

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

    def extract(self, df: pd.DataFrame) -> 'OrderedString':
        if self._categories is None:
            if isinstance(df[self.name].dtype, pd.CategoricalDtype) and df[self.name].cat.ordered:
                self.categories = df[self.name].cat.categories.to_list()
            else:
                raise ExtractionError
        super().extract(df=df)

        return self

    def merge_ordered_categories(self, ordered_cats1, ordered_cats2):
        """Merges two ordered category lists
        Ex: cats1 = [a,b,c,d,f], cats2 = [n,b,e,d,q]
        merged_cats = [a,n,b,c,e,d,q,f]"""

        if len(ordered_cats1) == 0:
            return ordered_cats2
        elif len(ordered_cats2) == 0:
            return ordered_cats1

        merged_cats = []
        cat2_start_idx = 0
        for cat in ordered_cats1:
            if cat in ordered_cats2:
                cat2_end_idx = ordered_cats2.index(cat)
                merged_cats += ordered_cats2[cat2_start_idx:cat2_end_idx]
                cat2_start_idx = cat2_end_idx + 1
            merged_cats.append(cat)

        merged_cats += ordered_cats2[cat2_start_idx:]
        return merged_cats

    def update_meta(self, df: pd.DataFrame) -> 'OrderedString':
        categories = []
        if isinstance(df[self.name].dtype, pd.CategoricalDtype) and df[self.name].cat.ordered is True:
            categories = df[self.name].cat.categories.to_list()

        self.categories = self.merge_ordered_categories(categories, self.categories)
        super().update_meta(df=df)

        return self

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        return d

    def less_than(self, x: str, y: str) -> bool:
        return cast(Sequence, self.categories).index(x) < cast(Sequence, self.categories).index(y)
