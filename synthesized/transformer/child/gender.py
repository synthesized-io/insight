from typing import List, Optional

import numpy as np
import pandas as pd

from ..base import Transformer
from ...config import GenderTransformerConfig
from ...metadata_new.model.person import GenderModel


class GenderTransformer(Transformer):
    """
    Transform a continuous distribution using the quantile transform.

    Attributes:
        name (str) : the data frame column to transform.
    """

    def __init__(self, name: str, gender_label: Optional[str] = None, title_label: Optional[str] = None,
                 config: GenderTransformerConfig = GenderTransformerConfig()):

        super().__init__(name=name)

        self.gender_label = gender_label
        self.title_label = title_label

        self.config = config
        self._fitted = True

    def __repr__(self):
        return (f'{self.__class__.__name__}(name="{self.name}", gender_label="{self.gender_label}", '
                f'title_label="{self.title_label}")')

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.gender_label is not None:
            df[self.name] = df[self.gender_label].astype(str).apply(self.config.get_gender_from_gender)
        elif self.title_label is not None:
            df[self.name] = df[self.title_label].astype(str).apply(self.config.get_gender_from_title)
        else:
            raise ValueError("Can't extract gender series as 'gender_label' nor 'title_label' are given.")

        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.gender_label is not None:
            df[self.gender_label] = df[self.name]
        if self.title_label is not None:
            df[self.title_label] = df[self.name].astype(dtype=str).apply(self.config.get_title_from_gender)

        return df

    @classmethod
    def from_meta(cls, meta: GenderModel) -> 'GenderTransformer':
        return cls(meta.name, gender_label=meta.gender_label, title_label=meta.title_label, config=meta.config)

    @property
    def in_columns(self) -> List[str]:
        in_columns = []
        if self.gender_label is not None:
            in_columns.append(self.gender_label)
        if self.title_label is not None:
            in_columns.append(self.title_label)
        return in_columns
