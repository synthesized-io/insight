from typing import Any, DefaultDict, Dict, Generic, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .histogram import Histogram
from ..base import DiscreteModel
from ..base.value_meta import NType
from ..value import AssociatedCategorical


class AssociatedHistogram(DiscreteModel[NType], Generic[NType], Mapping[str, DiscreteModel]):
    """
    A discrete model that represents an 'association' between two discrete models, this usually implies some
        structured rules between the two models. This class is mainly used by the value factory to extract
        associated categorical values.
    """

    def __init__(self, name: str, models: Sequence[DiscreteModel], binding_mask: np.ndarray = None,
                 categories_to_idx: Dict[str, DefaultDict[Any, int]] = None):
        super().__init__(name=name)
        self.children = models
        self.binding_mask = binding_mask
        self.categories_to_idx = categories_to_idx

    def fit(self, df: pd.DataFrame):
        for model in self.values():
            assert isinstance(model, DiscreteModel)
            model.fit(df)
        self._fitted = True

        return self

    @classmethod
    def from_meta(cls, association_meta: AssociatedCategorical):
        models = [Histogram.from_meta(meta) for meta in association_meta.values()]  # type: ignore
        assert hasattr(association_meta, "categories_to_idx")
        return cls(name=association_meta.name, models=models, binding_mask=association_meta.binding_mask,
                   categories_to_idx=association_meta.categories_to_idx)

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = self.children[0].sample(n, produce_nans, conditions)  # type: ignore

        for model in self.children[1:]:
            df = df.join(model.sample(n, produce_nans, conditions))  # type: ignore

        return df
