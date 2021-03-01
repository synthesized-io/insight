from typing import List, Optional

import numpy as np
import pandas as pd

from ..base import Transformer
from ...config import GenderTransformerConfig
from ...metadata_new.model.person import GenderModel
from ...util import get_gender_from_df, get_gender_title_from_df


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
        return get_gender_from_df(df, name=self.name, gender_label=self.gender_label,
                                  title_label=self.title_label, gender_mapping=self.config.gender_mapping,
                                  title_mapping=self.config.title_mapping)

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return get_gender_title_from_df(df, name=self.name, gender_label=self.gender_label,
                                        title_label=self.title_label, gender_mapping=self.config.gender_mapping,
                                        title_mapping=self.config.title_mapping)

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
