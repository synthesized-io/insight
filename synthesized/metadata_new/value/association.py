from collections import defaultdict
from typing import Any, DefaultDict, Dict, Sequence, cast

import numpy as np
import pandas as pd

from ..base import Nominal, ValueMeta
from ..exceptions import MetaNotExtractedError


class AssociatedCategorical(ValueMeta[Nominal, str]):
    dtype = 'O'

    def __init__(
        self, name: str, children: Sequence[Nominal], binding_mask: np.ndarray = None,
    ):
        super().__init__(name, children=children)
        self.binding_mask = binding_mask

    @property
    def categories_to_idx(self) -> Dict[str, DefaultDict[Any, int]]:
        if not self._extracted:
            raise MetaNotExtractedError

        return {
            model.name: defaultdict(int, {
                category: idx + 1 for idx, category in enumerate(cast(Sequence[Any], model.categories))
            }) for model in self.values()
        }

    def extract(self, df: pd.DataFrame):
        super().extract(df)
        if self.binding_mask is not None:
            return

        df_associated = df[self.keys()].copy()

        for name, idx_mapping in self.categories_to_idx.items():
            df_associated[name] = df_associated[name].map(idx_mapping).astype(int)

        counts = np.zeros(shape=[len(model.categories) + 1 for model in self.values()])  # type: ignore

        for _, row in df_associated.iterrows():
            idx = tuple(v for v in row.values)
            counts[idx] += 1

        self.binding_mask = (counts > 0).astype(np.int32)
        for axis in range(len(self.binding_mask.shape)):
            # remove the nan part of the binding mask (dealt with separately)
            self.binding_mask = np.delete(self.binding_mask, 0, axis)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'binding_mask': self.binding_mask
        })

        return d
