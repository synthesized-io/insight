from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .histogram import Histogram
from ..base import Model
from ...metadata_new.value import AssociatedCategorical


class AssociatedHistogram(AssociatedCategorical[Histogram], Model):
    """
    A discrete model that represents an 'association' between two discrete models, this usually implies some
        structured rules between the two models. This class is mainly used by the value factory to extract
        associated categorical values.
    """

    def __init__(self, name: str, models: Sequence[Histogram], binding_mask: np.ndarray = None):
        super().__init__(name=name, children=models, binding_mask=binding_mask)

    def fit(self, df: pd.DataFrame) -> 'AssociatedHistogram':
        for model in self.values():
            model.fit(df)
        return super().fit(df=df)

    @classmethod
    def from_meta(cls, association_meta: AssociatedCategorical) -> 'AssociatedHistogram':
        models = [Histogram.from_meta(meta) for meta in association_meta.values()]
        return cls(name=association_meta.name, models=models, binding_mask=association_meta.binding_mask)

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = self.children[0].sample(n, produce_nans, conditions)

        for model in self.children[1:]:
            df = df.join(model.sample(n, produce_nans, conditions))

        return df
