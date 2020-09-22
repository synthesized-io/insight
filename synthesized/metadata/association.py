import logging
from typing import List, Optional

import pandas as pd
import numpy as np
import tensorflow as tf

from .value_meta import ValueMeta
from .categorical import CategoricalMeta

logger = logging.getLogger(__name__)


class AssociationMeta(ValueMeta):
    def __init__(
            self, values: List[CategoricalMeta], associations: List[List[str]],
            binding_mask: Optional[np.ndarray] = None
    ):
        super(AssociationMeta, self).__init__(name='|'.join([v.name for v in values]))
        self.values = values
        logger.debug("Creating Associated value with associations: ")
        for n, association in enumerate(associations):
            logger.debug(f"{n + 1}: {association}")
        self.associations = associations
        self.dtype = tf.int64
        self.binding_mask: Optional[np.ndarray] = binding_mask

    def __str__(self) -> str:
        string = super().__str__()
        return string

    def columns(self) -> List[str]:
        return [name for value in self.values for name in value.columns()]

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)
        for v in self.values:
            v.extract(df=df)

        final_mask = np.ones(shape=[v.num_categories for v in self.values])

        for associated_values in self.associations:
            associated_values = list(associated_values)
            associated_values.sort(key=lambda x: [v.name for v in self.values].index(x))
            df2 = df[associated_values].copy()
            for v in self.values:
                df2[v.name] = df2[v.name].map(v.category2idx)

            a_dfmeta = {v.name: v for v in self.values}

            counts = np.zeros(shape=[a_dfmeta[v].num_categories for v in associated_values])

            for i, row in df2.iterrows():
                idx = tuple(v for v in row.values)
                counts[idx] += 1

            mask = (counts > 0).astype(np.int32)

            for n, v in enumerate(self.values):
                if v.name not in associated_values:
                    mask = np.expand_dims(mask, axis=n)

            final_mask *= np.broadcast_to(mask, [v.num_categories for v in self.values])

        self.binding_mask = final_mask

    def learned_input_columns(self) -> List[str]:
        return [col for value in self.values for col in value.learned_input_columns()]

    def learned_output_columns(self) -> List[str]:
        return [col for value in self.values for col in value.learned_output_columns()]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for value in self.values:
            df = value.preprocess(df)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        for value in self.values:
            df = value.postprocess(df)
        return df
