from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from ..base import Nominal, ValueMeta


class AssociatedCategorical(ValueMeta):
    dtype = 'O'

    def __init__(
        self, name: str, associated_metas: Sequence[Nominal], binding_mask: np.ndarray = None,
    ):
        super().__init__(name)
        self.children = associated_metas
        self.binding_mask = binding_mask

    def extract(self, df: pd.DataFrame):
        super().extract(df)
        if self.binding_mask is not None:
            return

        df_associated = df[self.keys()].copy()
        self.categories_to_idx = {model.name: defaultdict(int, {category: idx + 1 for idx, category
                                                                in enumerate(model.categories)}) for model in self.values()}  # type: ignore

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
