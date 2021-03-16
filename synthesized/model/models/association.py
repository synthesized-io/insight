from typing import Any, DefaultDict, Dict, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .histogram import Histogram
from ..base import Model
from ...metadata_new.value import AssociatedCategorical


class AssociatedHistogram(Model[AssociatedCategorical], Mapping[str, Histogram]):
    """
    A discrete model that represents an 'association' between two discrete models, this usually implies some
        structured rules between the two models. This class is mainly used by the value factory to extract
        associated categorical values.
    """

    def __init__(self, meta: AssociatedCategorical, models: Optional[Sequence[Histogram]] = None):
        models = [Histogram(meta) for meta in meta.values()] if models is None else models  # type: ignore
        super().__init__(meta=meta)
        self._children = {m.name: m for m in models} if models is not None else {}

    @property
    def children(self) -> Sequence[Histogram]:
        return [m for m in self._children.values()]

    def __getitem__(self, k: str) -> Histogram:
        return self._children[k]

    def __iter__(self) -> Iterator[str]:
        for key in self._children:
            yield key

    def __len__(self) -> int:
        return len(self._children)

    @property
    def binding_mask(self) -> np.ndarray:
        return self._meta.binding_mask

    @property
    def categories_to_idx(self) -> Dict[str, DefaultDict[int, Any]]:
        return self._meta.categories_to_idx

    def fit(self, df: pd.DataFrame):
        for model in self.children:
            model.fit(df)
        self._fitted = True

        return self

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = self.children[0].sample(n, produce_nans, conditions)  # type: ignore

        for model in self.children[1:]:
            df = df.join(model.sample(n, produce_nans, conditions))  # type: ignore

        return df
